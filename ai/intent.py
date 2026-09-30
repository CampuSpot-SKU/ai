"""의도 분류 — 신고 vs 행정문의 (명세서 4-3, 4-4).

역할 분담 (CAPD 원칙: "숫자 계산은 Python, 의미 해석은 Gemini")
- Gemini: 발화의 의미를 읽고 report_score / inquiry_score (0~100) + 플래그 2개만 JSON으로 출력
- Python: 점수 차가 AMBIGUITY_MARGIN(20점) 이내면 "애매함" → 되묻기 문구 선택. 최종 판정은 여기서.

실패 대응 (명세서 11장): Gemini 호출 1회 재시도 → 그래도 실패하면 IntentClassificationError.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# 두 점수 차가 이 값 이하이면 애매함으로 보고 되묻기 (명세서 4-4 확정값)
AMBIGUITY_MARGIN = 20
# 맥락으로 넘길 직전 대화 개수 (사용자/챗봇 메시지 합산)
MAX_HISTORY = 3
# 분류는 단순 작업이라 가장 빠르고 저렴한 Flash-Lite 사용 (2026-09-28 결정). 환경변수로 교체 가능.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

CLARIFY_REPORT_OR_INQUIRY = "이걸 신고로 접수해드릴까요, 안내가 필요하신 건가요?"
CLARIFY_VAGUE = "무엇에 대해 말씀하시는 건지 조금 더 알려주시겠어요?"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "intent_classification.md"

# chitchat = 인사·잡담, off_topic = 캠퍼스 시설 신고·학교 행정 문의와 무관한 요청 (둘 다 backend가 고정 문구 풀에서 짧게 답함)
Intent = Literal["report", "inquiry", "unclear", "chitchat", "off_topic"]
# 신고·문의가 아닌 말의 종류 — none이면 신고/문의/애매함 판정을 따름
Talk = Literal["none", "greeting", "thanks", "bye", "smalltalk", "about", "off_topic"]


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GeminiScores(BaseModel):
    """Gemini가 출력해야 하는 JSON 형식 (response_schema로 강제)."""

    report_score: int = Field(ge=0, le=100)
    inquiry_score: int = Field(ge=0, le=100)
    is_vague: bool = False
    safety_concern: bool = False
    talk: Talk = "none"


class IntentResult(BaseModel):
    intent: Intent
    report_score: int
    inquiry_score: int
    safety_concern: bool
    clarifying_question: str | None = None
    talk: Talk = "none"  # chitchat·off_topic일 때 어떤 말이었는지 (greeting/thanks/bye/smalltalk/about/off_topic)


class IntentClassificationError(Exception):
    """Gemini 호출이 재시도 후에도 실패한 경우."""


def decide(scores: GeminiScores) -> IntentResult:
    """점수 → 최종 판정. Gemini 호출 없이 순수 계산이라 유닛테스트 대상."""
    if scores.talk != "none":  # 신고·문의가 아닌 인사·잡담·범위 밖 — 점수는 무시
        return IntentResult(
            intent="off_topic" if scores.talk == "off_topic" else "chitchat",
            report_score=scores.report_score,
            inquiry_score=scores.inquiry_score,
            safety_concern=False,
            clarifying_question=None,
            talk=scores.talk,
        )
    diff = abs(scores.report_score - scores.inquiry_score)
    if scores.is_vague or diff <= AMBIGUITY_MARGIN:
        question = CLARIFY_VAGUE if scores.is_vague else CLARIFY_REPORT_OR_INQUIRY
        intent: Intent = "unclear"
    else:
        question = None
        intent = "report" if scores.report_score > scores.inquiry_score else "inquiry"
    return IntentResult(
        intent=intent,
        report_score=scores.report_score,
        inquiry_score=scores.inquiry_score,
        safety_concern=scores.safety_concern,
        clarifying_question=question,
    )


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


@lru_cache
def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_user_content(text: str, history: list[HistoryMessage]) -> str:
    lines = []
    recent = history[-MAX_HISTORY:]
    if recent:
        lines.append("## 이전 대화")
        for m in recent:
            speaker = "학생" if m.role == "user" else "챗봇"
            lines.append(f"{speaker}: {m.content}")
    else:
        lines.append("## 이전 대화\n(없음)")
    lines.append(f"\n## 분류할 마지막 발화\n{text}")
    return "\n".join(lines)


def _call_gemini(text: str, history: list[HistoryMessage]) -> GeminiScores:
    response = _client().models.generate_content(
        model=os.environ.get("GEMINI_INTENT_MODEL") or DEFAULT_MODEL,
        contents=_build_user_content(text, history),
        config=types.GenerateContentConfig(
            system_instruction=_system_prompt(),
            response_mime_type="application/json",
            response_schema=GeminiScores,
            temperature=0,
        ),
    )
    return GeminiScores.model_validate_json(response.text or "")


def classify(text: str, history: list[HistoryMessage] | None = None) -> IntentResult:
    history = history or []
    last_error: Exception | None = None
    for attempt in (1, 2):  # 1회 재시도
        try:
            return decide(_call_gemini(text, history))
        except (ValidationError, ValueError) as e:  # JSON 형식이 깨진 응답
            last_error = e
        except Exception as e:  # noqa: BLE001 — 네트워크/쿼터 등 Gemini 호출 오류 전부
            last_error = e
        logger.warning("intent classify attempt %d failed: %r", attempt, last_error)
    raise IntentClassificationError(str(last_error)) from last_error
