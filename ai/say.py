"""신고 접수 대화 문장 다듬기 — 챗봇이 "사람처럼" 말하게 하는 역할 (작업 1-3c 대화형 개편).

역할 분담: 무엇을 물을지·어느 단계인지는 backend 코드가 정하고, 여기서는 그 뜻을 그대로 둔 채
말투만 자연스럽게 바꾼다 (Gemini는 문장만 만든다). 그래서 매번 표현이 조금씩 달라도 된다.
안전장치: 결과에 must_include의 모든 문구가 들어 있어야 하고 길이·형식 검사를 통과해야 한다.
통과 못 하거나 Gemini가 실패하면 SayError → backend가 고정 문구를 쓴다 (명세서 11장 실패 대응).
"""
import logging
import os
from functools import lru_cache
from pathlib import Path

from google.genai import types
from pydantic import BaseModel, Field

from ai.intent import _client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_LEN = 300
_PROMPT_PATH = Path(__file__).parent / "prompts" / "report_say.md"


class SayError(Exception):
    """문장 생성 실패 또는 검사 탈락."""


class SayHistory(BaseModel):
    role: str
    content: str


class SayRequest(BaseModel):
    kind: str = Field(max_length=40, examples=["offer"])
    base_text: str = Field(min_length=1, max_length=400, description="전달해야 할 뜻이 담긴 기본 문장")
    must_include: list[str] = Field(default_factory=list, max_length=12)
    history: list[SayHistory] = Field(default_factory=list)


class SayResult(BaseModel):
    text: str


@lru_cache
def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_content(req: SayRequest) -> str:
    lines = ["## 이전 대화"]
    recent = req.history[-4:]
    if recent:
        for m in recent:
            lines.append(f"{'학생' if m.role == 'user' else '챗봇'}: {m.content}")
    else:
        lines.append("(없음)")
    lines.append(f"\n## 단계\n{req.kind}")
    lines.append(f"\n## 전달할 뜻 (기본 문장)\n{req.base_text}")
    if req.must_include:
        lines.append("\n## 반드시 그대로 들어가야 하는 표현\n" + "\n".join(f"- {s}" for s in req.must_include))
    return "\n".join(lines)


def validate(text: str, must_include: list[str]) -> str:
    """Gemini 결과 검사 — 통과하면 다듬은 문장, 아니면 SayError."""
    out = text.strip().strip('"').strip()
    if not out or len(out) > MAX_LEN:
        raise SayError("길이")
    if any(mark in out for mark in ("•", "·", "- ", "위치:", "상황:", "**", "#")):
        raise SayError("목록·라벨 형식")
    for needed in must_include:
        if needed not in out:
            raise SayError(f"필수 표현 누락: {needed}")
    return out


def _call_gemini(req: SayRequest) -> str:
    response = _client().models.generate_content(
        model=os.environ.get("GEMINI_SAY_MODEL") or os.environ.get("GEMINI_INTENT_MODEL") or DEFAULT_MODEL,
        contents=_build_content(req),
        config=types.GenerateContentConfig(
            system_instruction=_system_prompt(),
            temperature=0.8,
            max_output_tokens=200,
        ),
    )
    return response.text or ""


def say(req: SayRequest) -> SayResult:
    """기본 문장을 같은 뜻의 자연스러운 말로. 실패·검사 탈락 시 1회 재시도 후 SayError."""
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            return SayResult(text=validate(_call_gemini(req), req.must_include))
        except SayError as e:
            last = e
        except Exception as e:  # noqa: BLE001 — 네트워크/쿼터 등 Gemini 호출 오류 전부
            last = e
        logger.warning("report say attempt %d failed: %r", attempt, last)
    raise SayError(str(last)) from last
