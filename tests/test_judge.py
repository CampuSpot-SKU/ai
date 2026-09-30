"""신고 판정 유닛테스트 — Gemini를 부르지 않고 검사 규칙·재시도만 확인 (CI용)."""
import pytest

from ai import judge as judge_mod
from ai.judge import GeminiJudgement, JudgeError, JudgeRequest, judge, validate

CATS = ["전기", "시설·설비", "청소·위생", "안전", "IT·네트워크", "기타"]


def _req(text: str = "3층 화장실 물이 새요", location: str | None = "혜인관 3층 화장실") -> JudgeRequest:
    return JudgeRequest(text=text, location=location, categories=CATS)


def _g(**kw: object) -> GeminiJudgement:
    base: dict[str, object] = {
        "category": "시설·설비",
        "impact": "high",
        "urgency": "low",
        "problem_stated": True,
        "reason": "여러 학생이 쓰는 화장실이지만 당장 위험하지는 않아 보여요.",
    }
    base.update(kw)
    return GeminiJudgement.model_validate(base)


def test_validate_passes() -> None:
    out = validate(_g(), _req())
    assert out.category == "시설·설비" and out.impact == "high" and out.urgency == "low"


def test_validate_rejects_unknown_category() -> None:
    with pytest.raises(JudgeError):
        validate(_g(category="건축"), _req())


@pytest.mark.parametrize("reason", ["", "x" * 120, "P2로 봤어요.", "**위험**해요."])
def test_validate_rejects_bad_reason(reason: str) -> None:
    with pytest.raises(JudgeError):
        validate(_g(reason=reason), _req())


def test_validate_rejects_invented_place() -> None:
    with pytest.raises(JudgeError):
        validate(_g(reason="북악관 5층 복도가 위험해 보여요."), _req())


def test_judge_retries_once_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def boom(_req: JudgeRequest) -> GeminiJudgement:
        calls.append(1)
        raise RuntimeError("quota")

    monkeypatch.setattr(judge_mod, "_call_gemini", boom)
    with pytest.raises(JudgeError):
        judge(_req())
    assert len(calls) == 2


def test_judge_second_attempt_can_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([_g(category="없는것"), _g()])
    monkeypatch.setattr(judge_mod, "_call_gemini", lambda _r: next(results))
    assert judge(_req()).category == "시설·설비"
