"""탐지 배치 유닛테스트 — DB 없이 묶기·후보 계획 규칙만 확인 (1-8, 명세 3-3)."""
from datetime import UTC, datetime, timedelta

import pytest

from ai.detection import (
    ExistingCluster,
    ReportPoint,
    find_bursts,
    plan_detection,
)

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def _t(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def _p(rid: str, hours: float, building: str = "b1", detail: str = "화장실", cat: str = "c1"):
    return ReportPoint(rid, building, detail, cat, _t(hours))


# ---------------------------------------------------------------- find_bursts


def test_three_within_window_is_a_burst() -> None:
    items = [(_t(0), "a"), (_t(10), "b"), (_t(70), "c")]
    assert find_bursts(items, 3, 72) == [["a", "b", "c"]]


def test_two_reports_are_not_enough() -> None:
    assert find_bursts([(_t(0), "a"), (_t(1), "b")], 3, 72) == []


def test_window_boundary_is_inclusive() -> None:
    assert find_bursts([(_t(0), "a"), (_t(36), "b"), (_t(72), "c")], 3, 72) == [["a", "b", "c"]]
    assert find_bursts([(_t(0), "a"), (_t(36), "b"), (_t(72.01), "c")], 3, 72) == []


def test_input_order_does_not_matter() -> None:
    items = [(_t(70), "c"), (_t(0), "a"), (_t(10), "b")]
    assert find_bursts(items, 3, 72) == [["a", "b", "c"]]


def test_overlapping_windows_merge_into_one_burst() -> None:
    # 0·10·20h 와 10·20·80h 처럼 창이 겹치면 한 묶음
    items = [(_t(0), "a"), (_t(10), "b"), (_t(20), "c"), (_t(80), "d")]
    assert find_bursts(items, 3, 72) == [["a", "b", "c", "d"]]


def test_far_apart_bursts_stay_separate() -> None:
    items = [(_t(0), "a"), (_t(1), "b"), (_t(2), "c"), (_t(500), "d"), (_t(501), "e"), (_t(502), "f")]
    assert find_bursts(items, 3, 72) == [["a", "b", "c"], ["d", "e", "f"]]


def test_thresholds_are_parameters() -> None:
    items = [(_t(0), "a"), (_t(5), "b")]
    assert find_bursts(items, 2, 6) == [["a", "b"]]
    assert find_bursts(items, 2, 4) == []


def test_count_must_be_positive() -> None:
    with pytest.raises(ValueError):
        find_bursts([], 0, 72)


# ---------------------------------------------------------------- plan_detection


def test_new_cluster_for_a_burst() -> None:
    points = [_p("a", 0), _p("b", 1), _p("c", 2)]
    plan = plan_detection(points, [])
    assert plan.new_clusters == [(("b1", "화장실", "c1"), ["a", "b", "c"])]
    assert plan.additions == []


def test_different_place_or_category_is_not_grouped_together() -> None:
    points = [
        _p("a", 0),
        _p("b", 1, building="b2"),
        _p("c", 2, detail="복도"),
        _p("d", 3, cat="c2"),
    ]
    plan = plan_detection(points, [])
    assert plan.new_clusters == []


def test_two_places_make_two_clusters() -> None:
    points = [_p("a", 0), _p("b", 1), _p("c", 2)] + [
        _p("d", 0, building="b2"),
        _p("e", 1, building="b2"),
        _p("f", 2, building="b2"),
    ]
    plan = plan_detection(points, [])
    assert {key for key, _ in plan.new_clusters} == {
        ("b1", "화장실", "c1"),
        ("b2", "화장실", "c1"),
    }


def test_rerun_with_same_data_changes_nothing() -> None:
    points = [_p("a", 0), _p("b", 1), _p("c", 2)]
    existing = [ExistingCluster("k1", ("b1", "화장실", "c1"), "후보", frozenset({"a", "b", "c"}))]
    plan = plan_detection(points, existing)
    assert plan.new_clusters == []
    assert plan.additions == []


def test_new_report_extends_open_candidate() -> None:
    points = [_p("a", 0), _p("b", 1), _p("c", 2), _p("d", 30)]
    existing = [ExistingCluster("k1", ("b1", "화장실", "c1"), "후보", frozenset({"a", "b", "c"}))]
    plan = plan_detection(points, existing)
    assert plan.new_clusters == []
    assert plan.additions == [("k1", ["d"])]


def test_decided_cluster_reports_are_not_counted_again() -> None:
    # a·b·c 는 이미 승격됨. 새 신고 d·e 두 건만으로는 후보가 되지 않는다.
    points = [_p("a", 0), _p("b", 1), _p("c", 2), _p("d", 30), _p("e", 31)]
    existing = [ExistingCluster("k1", ("b1", "화장실", "c1"), "승격", frozenset({"a", "b", "c"}))]
    assert plan_detection(points, existing).new_clusters == []


def test_new_burst_after_rejection_creates_new_candidate() -> None:
    points = [_p("a", 0), _p("b", 1), _p("c", 2)] + [_p("d", 500), _p("e", 501), _p("f", 502)]
    existing = [ExistingCluster("k1", ("b1", "화장실", "c1"), "기각", frozenset({"a", "b", "c"}))]
    plan = plan_detection(points, existing)
    assert plan.new_clusters == [(("b1", "화장실", "c1"), ["d", "e", "f"])]


def test_separate_bursts_in_one_place_share_one_open_candidate() -> None:
    points = [_p("a", 0), _p("b", 1), _p("c", 2)] + [_p("d", 500), _p("e", 501), _p("f", 502)]
    plan = plan_detection(points, [])
    assert plan.new_clusters == [(("b1", "화장실", "c1"), ["a", "b", "c", "d", "e", "f"])]


def test_thresholds_from_settings_are_used() -> None:
    points = [_p("a", 0), _p("b", 1)]
    assert plan_detection(points, [], count=2, hours=24).new_clusters == [
        (("b1", "화장실", "c1"), ["a", "b"])
    ]
