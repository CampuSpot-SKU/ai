"""신고 판정 — 카테고리·영향도·긴급도·판정 이유 (작업 1-3b, 명세서 3-1·4-3).

역할 분담 (CAPD 원칙: "숫자 계산은 Python, 의미 해석은 Gemini")
- Gemini: 신고 문장을 읽고 카테고리, 영향도(높음/낮음), 긴급도(높음/낮음), 문제 내용을 말했는지,
  판정 이유 한 줄만 JSON으로 출력한다.
- Python(ai): 카테고리가 허용 목록 안인지, 이유 문장이 짧고 지어낸 건물·층·호수가 없는지 검사한다.
  우선순위(P1~P4)는 여기서 계산하지 않는다 — backend가 설정 테이블의 매트릭스로 정한다.

실패 대응 (명세서 11장): Gemini 호출 1회 재시도 → 그래도 실패하거나 검사에 탈락하면 JudgeError →
backend가 기존 규칙 기반 판정으로 대체한다.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from ai.intent import _client
from ai.say import _FACT_RE, _invented_fact  # noqa: F401 — 지어낸 건물·층·호수 검사 재사용

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_REASON_LEN = 90
FALLBACK_CATEGORY = "기타"
_PROMPT_PATH = Path(__file__).parent / "prompts" / "report_judge.md"

Level = Literal["high", "low"]


class JudgeError(Exception):
    """판정 호출이 재시도 후에도 실패했거나 검사에 탈락한 경우."""


class JudgeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, description="학생이 신고하며 말한 내용 (여러 메시지를 합친 것)")
    location: str | None = Field(
        default=None, max_length=100, description="backend가 학교 데이터로 확인한 위치 (예: '혜인관 2층 화장실'). 모르면 없음"
    )
    categories: list[str] = Field(min_length=1, max_length=20, description="고를 수 있는 카테고리 이름들")


class GeminiJudgement(BaseModel):
    """Gemini가 출력해야 하는 JSON 형식 (response_schema로 강제)."""

    category: str
    impact: Level
    urgency: Level
    problem_stated: bool
    reason: str


class JudgeResult(BaseModel):
    category: str
    impact: Level
    urgency: Level
    problem_stated: bool
    reason: str


@lru_cache
def _system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_content(req: JudgeRequest) -> str:
    lines = [
        "## 고를 수 있는 카테고리",
        ", ".join(req.categories),
        "\n## 확인된 위치",
        req.location or "(모름)",
        "\n## 학생이 말한 내용",
        req.text,
    ]
    return "\n".join(lines)


def validate(g: GeminiJudgement, req: JudgeRequest) -> JudgeResult:
    """Gemini 결과 검사 — 통과하면 JudgeResult, 아니면 JudgeError."""
    category = g.category.strip()
    if category not in req.categories:
        raise JudgeError(f"허용 목록에 없는 카테고리: {category}")
    reason = " ".join(g.reason.split()).strip('"').strip()
    if not reason or len(reason) > MAX_REASON_LEN:
        raise JudgeError("이유 길이")
    if any(mark in reason for mark in ("P1", "P2", "P3", "P4", "**", "#")):
        raise JudgeError("이유에 우선순위 등급·서식 포함")  # 등급은 backend 매트릭스가 정함
    if invented := _invented_fact(reason, f"{req.text} {req.location or ''}"):
        raise JudgeError(f"지어낸 표현: {invented}")
    return JudgeResult(
        category=category,
        impact=g.impact,
        urgency=g.urgency,
        problem_stated=g.problem_stated,
        reason=reason,
    )


def _call_gemini(req: JudgeRequest) -> GeminiJudgement:
    response = _client().models.generate_content(
        model=os.environ.get("GEMINI_JUDGE_MODEL") or os.environ.get("GEMINI_INTENT_MODEL") or DEFAULT_MODEL,
        contents=_build_content(req),
        config=types.GenerateContentConfig(
            system_instruction=_system_prompt(),
            response_mime_type="application/json",
            response_schema=GeminiJudgement,
            temperature=0,
        ),
    )
    return GeminiJudgement.model_validate_json(response.text or "")


def judge(req: JudgeRequest) -> JudgeResult:
    last_error: Exception | None = None
    for attempt in (1, 2):  # 1회 재시도
        try:
            return validate(_call_gemini(req), req)
        except JudgeError as e:
            last_error = e
        except (ValidationError, ValueError) as e:  # JSON 형식이 깨진 응답
            last_error = e
        except Exception as e:  # noqa: BLE001 — 네트워크/쿼터 등 Gemini 호출 오류 전부
            last_error = e
        logger.warning("report judge attempt %d failed: %r", attempt, last_error)
    raise JudgeError(str(last_error)) from last_error
