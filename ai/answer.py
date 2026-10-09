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
from collections.abc import Callable
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


def answer(question: str, *, session: Any, searcher: Searcher | None = None) -> AnswerResult:
    """질문 하나에 근거 기반 답변. 근거가 없으면 안내 문구 + sources=[] (Gemini 호출 없음)."""
    search = searcher or rag.search_hybrid
    hits = select_hits(search(question, session=session, k=TOP_K))
    if not hits:
        return AnswerResult(answer=NO_EVIDENCE_ANSWER, sources=[])
    last_error: Exception | None = None
    for attempt in (1, 2):  # 1회 재시도
        try:
            return validate(_call_gemini(question, hits), hits)
        except AnswerError as e:
            last_error = e
        except (ValidationError, ValueError) as e:  # JSON 형식이 깨진 응답
            last_error = e
        except Exception as e:  # noqa: BLE001 — 네트워크/쿼터 등 Gemini 호출 오류 전부
            last_error = e
        logger.warning("rag answer attempt %d failed: %r", attempt, last_error)
    raise AnswerError(str(last_error)) from last_error
