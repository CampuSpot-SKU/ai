"""의도 분류 유닛테스트 — Gemini를 실제로 부르지 않고 판정 규칙/재시도만 검사 (CI용).

실제 Gemini 정확도는 scripts/eval_intent.py (GitHub Actions 'Intent eval' 수동 실행)로 확인.
"""
import pytest

from ai import intent
from ai.intent import (
    CLARIFY_REPORT_OR_INQUIRY,
    CLARIFY_VAGUE,
    GeminiScores,
    HistoryMessage,
    IntentClassificationError,
    classify,
    decide,
)


@pytest.mark.parametrize(
    ("report", "inquiry", "expected"),
    [
        (95, 5, "report"),
        (3, 97, "inquiry"),
        (60, 40, "unclear"),  # 차이 20 → 경계값은 애매함
        (61, 39, "report"),  # 차이 22 → 판정
        (55, 45, "unclear"),
        (40, 61, "inquiry"),
    ],
)
def test_decide_by_margin(report: int, inquiry: int, expected: str) -> None:
    r = decide(GeminiScores(report_score=report, inquiry_score=inquiry))
    assert r.intent == expected
    assert (r.clarifying_question is None) == (expected != "unclear")


def test_unclear_mixed_asks_report_or_inquiry() -> None:
    r = decide(GeminiScores(report_score=58, inquiry_score=42))
    assert r.clarifying_question == CLARIFY_REPORT_OR_INQUIRY


def test_vague_overrides_scores() -> None:
    r = decide(GeminiScores(report_score=90, inquiry_score=10, is_vague=True))
    assert r.intent == "unclear"
    assert r.clarifying_question == CLARIFY_VAGUE


def test_safety_flag_passes_through() -> None:
    r = decide(GeminiScores(report_score=85, inquiry_score=15, safety_concern=True))
    assert r.intent == "report" and r.safety_concern is True


def test_history_is_trimmed_to_recent() -> None:
    history = [HistoryMessage(role="user", content=f"m{i}") for i in range(5)]
    content = intent._build_user_content("지금 발화", history)
    assert "m0" not in content and "m1" not in content and "m4" in content


def test_retry_once_then_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(text: str, history: list[HistoryMessage]) -> GeminiScores:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("temporary")
        return GeminiScores(report_score=95, inquiry_score=5)

    monkeypatch.setattr(intent, "_call_gemini", flaky)
    assert classify("물이 새요").intent == "report"
    assert calls["n"] == 2


def test_fails_after_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(text: str, history: list[HistoryMessage]) -> GeminiScores:
        raise ValueError("bad json")

    monkeypatch.setattr(intent, "_call_gemini", broken)
    with pytest.raises(IntentClassificationError):
        classify("물이 새요")


@pytest.mark.parametrize(
    ("talk", "expected"),
    [("greeting", "chitchat"), ("thanks", "chitchat"), ("bye", "chitchat"),
     ("smalltalk", "chitchat"), ("about", "chitchat"), ("off_topic", "off_topic")],
)
def test_talk_overrides_scores(talk: str, expected: str) -> None:
    """인사·잡담·범위 밖은 점수와 상관없이 해당 판정 (되묻기 문구 없음)."""
    r = decide(GeminiScores(report_score=50, inquiry_score=50, is_vague=True, talk=talk))  # type: ignore[arg-type]
    assert r.intent == expected and r.talk == talk and r.clarifying_question is None


def test_talk_none_keeps_normal_rules() -> None:
    assert decide(GeminiScores(report_score=95, inquiry_score=5)).intent == "report"
