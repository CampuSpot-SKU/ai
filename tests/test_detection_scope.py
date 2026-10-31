"""건물 전체 탐지 유닛테스트 — Gemini는 가짜로 바꿔 끼워서 묶기 규칙만 확인 (명세 3-3 확장)."""
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from ai.detection import WHOLE_BUILDING, ExistingCluster, ReportPoint, plan_detection
from ai.detection_scope import ScopeVerdict, find_building_wide

T0 = datetime(2026, 10, 30, 9, 0, tzinfo=UTC)
NAMES = {"b1": "유담관", "c1": "전기"}


def _p(rid: str, minutes: int, detail: str, building: str = "b1", cat: str = "c1") -> ReportPoint:
    return ReportPoint(rid, building, detail, cat, T0 + timedelta(minutes=minutes), None, f"{detail} 불이 나갔어요")


def _all_yes(building: str, category: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
    return ScopeVerdict(building_wide=True, report_numbers=list(range(1, len(cases) + 1)), reason="여러 호실 동시 정전")


def _no(building: str, category: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
    return ScopeVerdict(building_wide=False, report_numbers=[], reason="호실마다 따로")


ROOMS = [_p("a", 0, "101호"), _p("b", 5, "203호"), _p("c", 9, "305호"), _p("d", 14, "B101호")]


def test_building_wide_creates_one_cluster_with_whole_building_detail() -> None:
    plan = find_building_wide(ROOMS, [], 3, 72, _all_yes, NAMES)
    assert plan.calls == 1
    assert plan.new_clusters == [(("b1", WHOLE_BUILDING, "c1"), ["a", "b", "c", "d"])]
    assert plan.additions == []


def test_individual_faults_are_not_grouped() -> None:
    plan = find_building_wide(ROOMS, [], 3, 72, _no, NAMES)
    assert plan.new_clusters == [] and plan.additions == []


def test_same_detail_burst_is_left_to_normal_detection() -> None:
    same = [_p("a", 0, "101호"), _p("b", 5, "101호"), _p("c", 9, "101호")]
    called: list[int] = []

    def judge(b: str, c: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
        called.append(1)
        return _all_yes(b, c, cases)

    plan = find_building_wide(same, [], 3, 72, judge, NAMES)
    assert plan.calls == 0 and not called and plan.new_clusters == []


def test_other_building_or_category_is_not_mixed() -> None:
    pts = [_p("a", 0, "101호"), _p("b", 5, "203호", building="b2"), _p("c", 9, "305호", cat="c2")]
    assert find_building_wide(pts, [], 3, 72, _all_yes, NAMES).calls == 0


def test_too_few_reports_picked_means_no_cluster() -> None:
    def judge(b: str, c: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
        return ScopeVerdict(building_wide=True, report_numbers=[1, 2], reason="두 건만")

    assert find_building_wide(ROOMS, [], 3, 72, judge, NAMES).new_clusters == []


def test_picked_numbers_out_of_range_are_ignored() -> None:
    def judge(b: str, c: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
        return ScopeVerdict(building_wide=True, report_numbers=[0, 1, 2, 3, 99], reason="")

    plan = find_building_wide(ROOMS, [], 3, 72, judge, NAMES)
    assert plan.new_clusters == [(("b1", WHOLE_BUILDING, "c1"), ["a", "b", "c"])]


def test_gemini_failure_skips_without_breaking() -> None:
    def judge(b: str, c: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
        raise RuntimeError("Gemini 장애")

    plan = find_building_wide(ROOMS, [], 3, 72, judge, NAMES)
    assert plan.calls == 1 and plan.new_clusters == []


def test_call_limit() -> None:
    pts = []
    for i in range(4):  # 건물 4곳, 각각 호실 3개씩
        pts += [_p(f"{i}a", 0, "101호", building=f"x{i}"), _p(f"{i}b", 3, "202호", building=f"x{i}"), _p(f"{i}c", 6, "303호", building=f"x{i}")]
    plan = find_building_wide(pts, [], 3, 72, _all_yes, NAMES, max_calls=2)
    assert plan.calls == 2 and len(plan.new_clusters) == 2


def test_reports_already_in_a_whole_building_cluster_are_not_counted_again() -> None:
    done = ExistingCluster("k1", ("b1", WHOLE_BUILDING, "c1"), "기각", frozenset({"a", "b", "c", "d"}))
    assert find_building_wide(ROOMS, [done], 3, 72, _all_yes, NAMES).calls == 0


def test_open_whole_building_cluster_gets_new_reports_added() -> None:
    open_c = ExistingCluster("k1", ("b1", WHOLE_BUILDING, "c1"), "후보", frozenset({"old1"}))
    plan = find_building_wide(ROOMS, [open_c], 3, 72, _all_yes, NAMES)
    assert plan.new_clusters == [] and plan.additions == [("k1", ["a", "b", "c", "d"])]


def test_whole_building_cluster_is_ignored_by_place_detection() -> None:
    """건물 전체 후보가 있어도 같은 방 3건은 기존 규칙으로 자기 후보를 만든다."""
    whole = ExistingCluster("k1", ("b1", WHOLE_BUILDING, "c1"), "후보", frozenset({"x"}))
    pts = [_p("a", 0, "101호"), _p("b", 5, "101호"), _p("c", 9, "101호")]
    plan = plan_detection(pts, [whole])
    assert plan.new_clusters == [(("b1", "101호", "c1"), ["a", "b", "c"])]
    assert plan.additions == []


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_scope_can_be_disabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from ai.detection_scope import scope_enabled

    monkeypatch.setenv("DETECTION_SCOPE_AI", value)
    assert not scope_enabled()


def test_run_detection_puts_building_wide_cluster_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    from ai import detection, detection_scope

    applied: list[Any] = []
    monkeypatch.delenv("DETECTION_SCOPE_AI", raising=False)
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0")
    monkeypatch.setattr(detection, "load_thresholds", lambda db: (3, 72))
    monkeypatch.setattr(detection, "load_points", lambda db: list(ROOMS))
    monkeypatch.setattr(detection, "load_existing", lambda db: [])
    monkeypatch.setattr(detection, "load_names", lambda db: NAMES)
    monkeypatch.setattr(detection, "apply_plan", lambda db, plan: applied.append(plan))
    monkeypatch.setattr(detection_scope, "gemini_scope_judge", _all_yes)
    result = detection.run_detection(object())
    assert result.building_wide_created == 1 and result.building_wide_calls == 1
    assert result.clusters_created == 1 and result.links_added == 4
    assert applied[0].new_clusters == [(("b1", WHOLE_BUILDING, "c1"), ["a", "b", "c", "d"])]


def test_run_detection_survives_scope_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai import detection

    monkeypatch.delenv("DETECTION_SCOPE_AI", raising=False)
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0")
    monkeypatch.setattr(detection, "load_thresholds", lambda db: (3, 72))
    monkeypatch.setattr(detection, "load_points", lambda db: list(ROOMS))
    monkeypatch.setattr(detection, "load_existing", lambda db: [])
    monkeypatch.setattr(detection, "apply_plan", lambda db, plan: None)

    def boom(db: object) -> dict[str, str]:
        raise RuntimeError("DB 오류")

    monkeypatch.setattr(detection, "load_names", boom)
    result = detection.run_detection(object())
    assert result.building_wide_created == 0 and result.clusters_created == 0
