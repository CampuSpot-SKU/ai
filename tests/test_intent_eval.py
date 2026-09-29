"""S-7a 의도 분류 평가 세트·비교군·지표 검사 (Gemini 호출 없음, CI용 — 명세서 10-4).

- 정답 세트 형식(열·판정 값)이 맞는지
- 평가 세트가 프롬프트 예시·개발 세트와 겹치지 않는지 ("처음 보는 문장" 평가가 깨지지 않게)
- 규칙 기반 비교군·지표 계산이 예상대로 동작하는지
"""

import pytest
from baseline_intent_rules import classify_rules
from intent_eval_lib import (
    LABELS,
    PROMPT_PATH,
    SETS,
    compute_metrics,
    find_overlaps,
    load_cases,
    prompt_examples,
)


@pytest.mark.parametrize("set_name", sorted(SETS))
def test_golden_sets_are_valid(set_name: str) -> None:
    cases = load_cases(SETS[set_name])
    assert cases
    assert {c.expected for c in cases} == set(LABELS)


def test_holdout_composition() -> None:
    cases = load_cases(SETS["holdout"])
    assert len(cases) >= 90
    assert sum(bool(c.history_raw) for c in cases) >= 10  # 맥락 판단 문장 포함


def test_holdout_does_not_overlap_prompt_or_dev() -> None:
    refs = [c.text for c in load_cases(SETS["dev"])]
    refs += prompt_examples(PROMPT_PATH.read_text(encoding="utf-8"))
    hits = find_overlaps([c.text for c in load_cases(SETS["holdout"])], refs)
    assert hits == [], f"평가 세트 문장을 바꿀 것: {hits}"


def test_prompt_examples_are_extracted() -> None:
    examples = prompt_examples(PROMPT_PATH.read_text(encoding="utf-8"))
    assert "휴학 신청 어떻게 해요?" in examples
    assert "공학관이요" in examples  # 이전 대화가 있는 예시도 포함


def test_overlap_detects_near_duplicate() -> None:
    hits = find_overlaps(["3동 2층 화장실에 물이 계속 새요!"], ["3동 2층 화장실 물이 계속 새요"])
    assert len(hits) == 1


@pytest.mark.parametrize(
    ("text", "history", "expected"),
    [
        ("복도 조명이 깜빡거려요", [], "report"),
        ("휴학 신청 어떻게 해요?", [], "inquiry"),
        ("화장실 냄새가 너무 심해요 어떻게 해야하죠", [], "unclear"),
        ("저기요", [], "unclear"),
        ("2층이요", [("assistant", "몇 층인지 알려주시겠어요?")], "report"),
        (
            "안내해주세요",
            [("assistant", "이걸 신고로 접수해드릴까요 안내가 필요하신 건가요?")],
            "inquiry",
        ),
    ],
)
def test_rules_baseline(text: str, history: list[tuple[str, str]], expected: str) -> None:
    assert classify_rules(text, history) == expected


def test_metrics() -> None:
    m = compute_metrics(
        ["report", "report", "inquiry", "unclear"], ["report", "inquiry", "inquiry", "ERROR"]
    )
    assert m.accuracy == 0.5
    assert m.precision("inquiry") == 0.5
    assert m.recall("report") == 0.5
    assert m.confusion["unclear"]["ERROR"] == 1
