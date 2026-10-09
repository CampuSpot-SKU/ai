"""예측 계산 유닛테스트 — DB 없이 평균 재발 주기와 다음 예상 시점만 확인 (1-11, 명세 3-3)."""
from datetime import UTC, datetime, timedelta

from ai.prediction import build_stats, compute_stat

T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def _d(days: float) -> datetime:
    return T0 + timedelta(days=days)


def test_fewer_than_three_is_not_predictable() -> None:
    assert compute_stat([]) is None
    assert compute_stat([_d(0), _d(10)]) is None


def test_average_gap_and_next_date() -> None:
    result = compute_stat([_d(0), _d(10), _d(30)])  # 간격 10일, 20일 → 평균 15일
    assert result is not None
    avg, last, nxt = result
    assert avg == 15.0
    assert last == _d(30)
    assert nxt == _d(45)


def test_order_of_input_does_not_matter() -> None:
    result = compute_stat([_d(30), _d(0), _d(10)])
    assert result is not None
    assert result[0] == 15.0


def test_fractional_days_are_kept() -> None:
    result = compute_stat([_d(0), _d(1), _d(1.5)])  # 간격 1일, 0.5일 → 평균 0.75일
    assert result is not None
    assert result[0] == 0.75


def test_build_stats_groups_by_building_and_category() -> None:
    rows = [
        ("b1", "c1", _d(0)),
        ("b1", "c1", _d(10)),
        ("b1", "c1", _d(20)),
        ("b1", "c2", _d(0)),  # 다른 카테고리 — 1건뿐이라 제외
        ("b2", "c1", _d(5)),
        ("b2", "c1", _d(6)),  # 2건뿐이라 제외
    ]
    stats = build_stats(rows)
    assert len(stats) == 1
    s = stats[0]
    assert (s.building_id, s.category_id) == ("b1", "c1")
    assert s.avg_recurrence_days == 10.0
    assert s.last_occurred_at == _d(20)
    assert s.predicted_next_at == _d(30)


def test_build_stats_empty() -> None:
    assert build_stats([]) == []
