"""행정 문의 RAG 답변 — 학칙·안내·공지를 근거로 답하고 근거 목록(sources)을 함께 돌려준다 (작업 1-4c, 명세서 4-5·5-1).

역할 분담 (CAPD 원칙: "숫자 계산은 Python, 의미 해석은 Gemini")
- Python: 검색(벡터 + `제N조` 보조) → 너무 먼 청크 버리기 → 거의 같은 청크 합치기 → Gemini가 낸 근거 번호를
  sources(제목·조 번호·링크)로 바꾸기. 근거 번호가 범위 밖이거나 답변이 서식을 쓰면 탈락시킨다.
- Gemini: 자료를 읽고 4-5 스타일로 답변을 쓰고 어느 자료를 썼는지 번호로 알려준다.

근거가 없으면 지어내지 않고 안내 문구 + `sources: []`를 돌려준다(Gemini 호출 없이).
실패 대응 (명세서 11장): Gemini 호출 1회 재시도 → 그래도 실패하면 AnswerError → backend가 503 처리.
"""
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from ai import rag
from ai.intent import _client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
TOP_K = 5
# 코사인 거리가 이보다 크면 질문과 상관없는 자료로 보고 버린다. 평가 세트(S-7)로 조정할 값 — RAG_MAX_DISTANCE로 덮어쓸 수 있다.
DEFAULT_MAX_DISTANCE = 0.55
SIMILAR_RATIO = 0.9  # 앞부분이 이만큼 같으면 같은 내용으로 보고 하나만 쓴다(예: 휴학 안내 페이지 두 개)
SIMILAR_PREFIX = 400
MAX_ANSWER_LEN = 900
KST = timezone(timedelta(hours=9))
_PROMPT_PATH = Path(__file__).parent / "prompts" / "rag_answer.md"

NO_EVIDENCE_ANSWER = (
    "이 내용은 제가 가진 학교 자료에서 근거를 찾지 못했어요. "
    "정확한 안내는 학교 대표번호(02-940-7114)나 학교 홈페이지에서 확인해 주세요."
)


class AnswerError(Exception):
    """답변 생성이 재시도 후에도 실패했거나 검사에 탈락한 경우."""


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["휴학은 언제까지 신청해야 해요?"])


class Source(BaseModel):
    title: str
    article_no: str | None = None
    url: str | None = None
    as_of: str | None = None  # 학칙·규정은 기준일('2024.9.1 기준') — 일부만 개정되므로 칩에 같이 보여준다


class AnswerResult(BaseModel):
    answer: str
    sources: list[Source]


class GeminiAnswer(BaseModel):
    """Gemini가 출력해야 하는 JSON 형식 (response_schema로 강제)."""

    answerable: bool
    answer: str
    used: list[int]


Searcher = Callable[..., list[rag.Hit]]  # (질문, session=, k=) → 검색 결과 — 테스트에서 가짜로 바꿔 끼움


@lru_cache
def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def max_distance() -> float:
    raw = os.environ.get("RAG_MAX_DISTANCE", "")
    try:
        return float(raw) if raw else DEFAULT_MAX_DISTANCE
    except ValueError:
        return DEFAULT_MAX_DISTANCE


def select_hits(hits: list[rag.Hit], limit: float | None = None) -> list[rag.Hit]:
    """너무 먼 청크를 버리고, 거의 같은 청크는 먼저 나온 것 하나만 남긴다."""
    cutoff = max_distance() if limit is None else limit
    kept: list[rag.Hit] = []
    for h in hits:
        if h.distance > cutoff:
            continue
        head = h.chunk_text[:SIMILAR_PREFIX]
        if any(SequenceMatcher(None, head, k.chunk_text[:SIMILAR_PREFIX]).ratio() >= SIMILAR_RATIO for k in kept):
            continue
        kept.append(h)
    return kept


def _date_label(published_at: str | None) -> str:
    if not published_at:
        return ""
    when = datetime.fromisoformat(published_at)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(KST).strftime("%Y-%m-%d")


def _build_content(question: str, hits: list[rag.Hit]) -> str:
    lines = ["## 자료"]
    for i, h in enumerate(hits, 1):
        date = _date_label(h.published_at) if h.doc_type == rag.DOC_TYPE_NOTICE else ""
        meta = f"{h.doc_type}" + (f", 게시일 {date}" if date else "")
        lines.append(f"\n[{i}] 제목: {h.title} ({meta})\n{h.chunk_text}")
    lines.append(f"\n## 학생 질문\n{question}")
    return "\n".join(lines)


def _as_of(h: rag.Hit) -> str | None:
    if h.doc_type != rag.DOC_TYPE_REGULATION or not h.published_at:
        return None
    d = datetime.fromisoformat(h.published_at)
    return f"{d.year}.{d.month}.{d.day} 기준"


def _source(h: rag.Hit) -> Source:
    return Source(title=h.title, article_no=h.article_no, url=h.source_url, as_of=_as_of(h))


def validate(g: GeminiAnswer, hits: list[rag.Hit]) -> AnswerResult:
    """Gemini 결과 검사 — 통과하면 AnswerResult, 아니면 AnswerError."""
    if not g.answerable:
        return AnswerResult(answer=NO_EVIDENCE_ANSWER, sources=[])
    answer = g.answer.strip()
    if not answer or len(answer) > MAX_ANSWER_LEN:
        raise AnswerError("답변 길이")
    if any(mark in answer for mark in ("**", "##", "```")):
        raise AnswerError("답변에 마크다운 서식 포함")
    if not g.used or any(n < 1 or n > len(hits) for n in g.used):
        raise AnswerError("근거 번호 오류")
    sources: list[Source] = []
    for n in dict.fromkeys(g.used):  # 중복 제거, 순서 유지
        src = _source(hits[n - 1])
        if src not in sources:
            sources.append(src)
    return AnswerResult(answer=answer, sources=sources)


# ---------------------------------------------------------------- 규정 개정 공지 연결 (1-4f)

REVISION_SQL_PATTERN = "개정|사전 ?공[고지]|공포"  # 공지 제목에서 개정 관련 글을 찾는 DB 정규식
REVISION_NOTICE_LIMIT = 50


@dataclass(frozen=True)
class RevisionNotice:
    title: str
    url: str | None
    published_at: datetime


RevisionLoader = Callable[[Any], list[RevisionNotice]]


def load_revision_notices(session: Any) -> list[RevisionNotice]:
    """학교 공지 중 제목에 개정·사전공고·공포가 들어간 최근 글(최신순)."""
    from sqlalchemy import text

    rows = session.execute(
        text(
            "SELECT title, source_url, published_at FROM admin_reg_documents "
            "WHERE doc_type = CAST(:t AS doc_type) AND published_at IS NOT NULL "
            "AND title ~ :pat ORDER BY published_at DESC LIMIT :n"
        ),
        {"t": rag.DOC_TYPE_NOTICE, "pat": REVISION_SQL_PATTERN, "n": REVISION_NOTICE_LIMIT},
    )
    return [RevisionNotice(r.title, r.source_url, r.published_at) for r in rows]


def regulation_name(title: str) -> str:
    """"학칙 제29조(휴학)" → "학칙", "학생생활규정 제3조(…)" → "학생생활규정"."""
    return re.split(r"\s제\d", title, maxsplit=1)[0].strip()


def _mentions(notice_title: str, name: str) -> bool:
    flat_title, flat_name = notice_title.replace(" ", ""), name.replace(" ", "")
    if not flat_name or flat_name not in flat_title:
        return False
    # 대학원 학칙 개정 공지는 (대학) 학칙과 다른 규정이라 제외 — "대학·대학원 학칙"처럼 둘 다 다루는 제목은 통과
    return not (flat_name == "학칙" and flat_title.startswith("대학원학칙"))


def _aware(when: datetime) -> datetime:
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def pick_revision(used: list[rag.Hit], notices: list[RevisionNotice]) -> RevisionNotice | None:
    """답변 근거가 된 규정(학칙·규정집)의 기준일 이후에 올라온, 그 규정 이름이 제목에 있는 개정 공지 중 가장 최근 것."""
    best: RevisionNotice | None = None
    for h in used:
        if h.doc_type != rag.DOC_TYPE_REGULATION or not h.published_at:
            continue
        base = _aware(datetime.fromisoformat(h.published_at))
        name = regulation_name(h.title)
        for n in notices:
            newer = _aware(n.published_at) > base and _mentions(n.title, name)
            if newer and (best is None or n.published_at > best.published_at):
                best = n
    return best


def with_revision_notice(
    result: AnswerResult, used: list[rag.Hit], notices: list[RevisionNotice]
) -> AnswerResult:
    """근거가 규정인데 그 뒤에 개정 공지가 있으면 답변 끝에 안내 한 줄 + 그 공지를 근거 칩에 추가."""
    notice = pick_revision(used, notices)
    if notice is None:
        return result
    when = _aware(notice.published_at).astimezone(KST)
    note = (
        f"\n\n참고: 이 규정은 {when.month}월 {when.day}일에 개정 공지가 있었어요. "
        "최신 내용은 아래 공지에서 확인해 주세요."
    )
    sources = [
        *result.sources,
        Source(title=notice.title, url=notice.url, as_of=f"{when.year}.{when.month}.{when.day} 게시"),
    ]
    return AnswerResult(answer=result.answer + note, sources=sources)


def _call_gemini(question: str, hits: list[rag.Hit]) -> GeminiAnswer:
    response = _client().models.generate_content(
        model=os.environ.get("GEMINI_RAG_MODEL") or os.environ.get("GEMINI_INTENT_MODEL") or DEFAULT_MODEL,
        contents=_build_content(question, hits),
        config=types.GenerateContentConfig(
            system_instruction=_system_prompt(),
            response_mime_type="application/json",
            response_schema=GeminiAnswer,
            temperature=0,
        ),
    )
    return GeminiAnswer.model_validate_json(response.text or "")


def _decorate(
    result: AnswerResult,
    g: GeminiAnswer,
    hits: list[rag.Hit],
    session: Any,
    revisions: RevisionLoader | None,
) -> AnswerResult:
    if session is None and revisions is None:
        return result
    if not result.sources:
        return result
    try:
        notices = (revisions or load_revision_notices)(session)
        used = [hits[n - 1] for n in dict.fromkeys(g.used)]
        return with_revision_notice(result, used, notices)
    except Exception:
        logger.warning("revision notice lookup failed", exc_info=True)
        return result


def answer(
    question: str,
    *,
    session: Any,
    searcher: Searcher | None = None,
    revisions: RevisionLoader | None = None,
) -> AnswerResult:
    """질문 하나에 근거 기반 답변. 근거가 없으면 안내 문구 + sources=[] (Gemini 호출 없음).
    근거가 규정이고 그 뒤에 개정 공지가 있으면 안내 한 줄을 덧붙인다(1-4f). DB 조회 실패는 무시."""
    search = searcher or rag.search_hybrid
    hits = select_hits(search(question, session=session, k=TOP_K))
    if not hits:
        return AnswerResult(answer=NO_EVIDENCE_ANSWER, sources=[])
    last_error: Exception | None = None
    for attempt in (1, 2):  # 1회 재시도
        try:
            g = _call_gemini(question, hits)
            result = validate(g, hits)
            return _decorate(result, g, hits, session, revisions)
        except AnswerError as e:
            last_error = e
        except (ValidationError, ValueError) as e:  # JSON 형식이 깨진 응답
            last_error = e
        except Exception as e:  # noqa: BLE001 — 네트워크/쿼터 등 Gemini 호출 오류 전부
            last_error = e
        logger.warning("rag answer attempt %d failed: %r", attempt, last_error)
    raise AnswerError(str(last_error)) from last_error
