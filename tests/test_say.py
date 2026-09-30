"""문장 다듬기 유닛테스트 — Gemini를 부르지 않고 검사 규칙/재시도만 확인 (CI용)."""
import pytest

from ai import say as say_mod
from ai.say import SayError, SayRequest, say, validate


def test_validate_passes_natural_sentence() -> None:
    out = validate("정수기 고장 관련해서 접수를 도와드릴까요?", ["접수"])
    assert out.endswith("?")


@pytest.mark.parametrize(
    "bad",
    ["", "x" * 400, "· 위치: 혜인관 3층", "**접수** 할까요?", "위치: 혜인관"],
)
def test_validate_rejects_bad_format(bad: str) -> None:
    with pytest.raises(SayError):
        validate(bad, [])


def test_validate_requires_must_include() -> None:
    with pytest.raises(SayError):
        validate("어느 건물이에요?", ["은주1관"])


def _req() -> SayRequest:
    return SayRequest(kind="offer", base_text="접수를 도와드릴까요?", must_include=["접수"])


def test_say_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake(_req: SayRequest) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("quota")
        return "접수 도와드릴까요?"

    monkeypatch.setattr(say_mod, "_call_gemini", fake)
    assert say(_req()).text == "접수 도와드릴까요?"
    assert calls["n"] == 2


def test_say_fails_after_two_bad_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(say_mod, "_call_gemini", lambda _r: "다른 얘기예요")  # 필수 표현 누락
    with pytest.raises(SayError):
        say(_req())
