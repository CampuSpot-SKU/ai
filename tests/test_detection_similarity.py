"""탐지 임베딩 유사도 보조 유닛테스트 — DB·Gemini 없이 묶기 규칙과 임베딩 채우기 흐름만 확인 (1-8b, 명세 3-3)."""
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai import detection
from ai.detection import (
    DEFAULT_SIMILARITY_DISTANCE,
    ExistingCluster,
    ReportPoint,
    cosine_distance,
    embed_missing_reports,
    format_vector,
    group_key,
    group_points,
    parse_vector,
    plan_detection,
    run_detection,
    similarity_distance,
)

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)

LEAK_A = (1.0, 0.0, 0.0)
LEAK_B = (0.99, 0.1, 0.0)  # LEAK_A와 아주 가까움(거리 약 0.005)
LEAK_C = (0.95, 0.31, 0.0)  # LEAK_B와 가깝고 LEAK_A와는 조금 더 멂(연쇄 확인용)
OTHER = (0.0, 1.0, 0.0)  # 전혀 다른 내용


def _p(
    rid: str,
    hours: float,
    detail: str = "화장실",
    emb: tuple[float, ...] | None = None,
    building: str = "b1",
    cat: str = "c1",
) -> ReportPoint:
    return ReportPoint(rid, building, detail, cat, T0 + timedelta(hours=hours), emb)


# ---------------------------------------------------------------- 코사인 거리


def test_cosine_distance_basics() -> None:
    assert cosine_distance(LEAK_A, LEAK_A) == pytest.approx(0.0)
    assert cosine_distance(LEAK_A, OTHER) == pytest.approx(1.0)
    assert cosine_distance((2.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)  # 크기는 상관없음


def test_cosine_distance_zero_vector_is_farthest() -> None:
    assert cosine_distance((0.0, 0.0), LEAK_A[:2]) == 1.0


def test_cosine_distance_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cosine_distance((1.0, 0.0), (1.0, 0.0, 0.0))


# ---------------------------------------------------------------- 묶기


def test_different_detail_with_close_embedding_is_one_group() -> None:
    pts = [_p("a", 0, "화장실", LEAK_A), _p("b", 1, "남자화장실", LEAK_B)]
    groups = group_points(pts, 0.1)
    assert [sorted(p.id for p in g) for g in groups] == [["a", "b"]]


def test_without_max_distance_only_same_detail_is_grouped() -> None:
    pts = [_p("a", 0, "화장실", LEAK_A), _p("b", 1, "남자화장실", LEAK_B)]
    assert len(group_points(pts)) == 2
    assert len(group_points(pts, 0)) == 2  # 0 이하는 임베딩 보조 끔


def test_far_embedding_stays_separate() -> None:
    pts = [_p("a", 0, "화장실", LEAK_A), _p("b", 1, "복도", OTHER)]
    assert len(group_points(pts, 0.1)) == 2


def test_missing_embedding_never_merges_different_detail() -> None:
    pts = [_p("a", 0, "화장실", LEAK_A), _p("b", 1, "남자화장실", None)]
    assert len(group_points(pts, 0.5)) == 2


def test_same_detail_groups_even_without_embedding() -> None:
    pts = [_p("a", 0, "화장실"), _p("b", 1, "화장실")]
    assert len(group_points(pts, 0.1)) == 1


def test_never_merges_across_building_or_category() -> None:
    pts = [
        _p("a", 0, "화장실", LEAK_A),
        _p("b", 1, "남자화장실", LEAK_B, building="b2"),
        _p("c", 2, "남자화장실", LEAK_B, cat="c2"),
    ]
    assert len(group_points(pts, 0.5)) == 3


def test_chain_links_far_ends() -> None:
    a_c = cosine_distance(LEAK_A, LEAK_C)
    a_b = cosine_distance(LEAK_A, LEAK_B)
    b_c = cosine_distance(LEAK_B, LEAK_C)
    limit = max(a_b, b_c) + 1e-9
    assert a_c > limit  # A와 C는 직접 기준 밖이지만
    pts = [_p("a", 0, "화장실", LEAK_A), _p("b", 1, "남자화장실", LEAK_B), _p("c", 2, "세면대", LEAK_C)]
    assert len(group_points(pts, limit)) == 1  # B를 거쳐 한 묶음


def test_group_key_uses_most_common_detail_then_earliest() -> None:
    majority = [_p("a", 0, "화장실"), _p("b", 1, "남자화장실"), _p("c", 2, "남자화장실")]
    assert group_key(majority) == ("b1", "남자화장실", "c1")
    tie = [_p("a", 5, "화장실"), _p("b", 1, "남자화장실")]
    assert group_key(tie)[1] == "남자화장실"  # 같으면 먼저 접수된 표기


# ---------------------------------------------------------------- 계획


def _three_leaks() -> list[ReportPoint]:
    return [
        _p("a", 0, "3동 2층 화장실", LEAK_A),
        _p("b", 10, "남자화장실", LEAK_B),
        _p("c", 20, "화장실 세면대", LEAK_B),
    ]


def test_similar_reports_with_different_details_make_a_candidate() -> None:
    plan = plan_detection(_three_leaks(), [], max_distance=0.1)
    assert len(plan.new_clusters) == 1
    key, ids = plan.new_clusters[0]
    assert sorted(ids) == ["a", "b", "c"]
    assert key[0] == "b1" and key[2] == "c1"
    assert plan.groups == 1 and plan.merged_groups == 1


def test_same_reports_without_similarity_make_no_candidate() -> None:
    plan = plan_detection(_three_leaks(), [])
    assert plan.new_clusters == [] and plan.merged_groups == 0


def test_time_window_still_applies_to_similar_reports() -> None:
    pts = [
        _p("a", 0, "화장실", LEAK_A),
        _p("b", 100, "남자화장실", LEAK_B),
        _p("c", 200, "세면대", LEAK_B),
    ]
    assert plan_detection(pts, [], max_distance=0.1).new_clusters == []


def test_open_candidate_is_extended_even_if_representative_detail_changed() -> None:
    pts = [*_three_leaks(), _p("d", 30, "남자화장실", LEAK_B)]
    old = ExistingCluster("k1", ("b1", "3동 2층 화장실", "c1"), "후보", frozenset({"a", "b", "c"}))
    plan = plan_detection(pts, [old], max_distance=0.1)
    assert plan.new_clusters == []
    assert plan.additions == [("k1", ["d"])]


def test_rerun_with_similarity_changes_nothing() -> None:
    first = plan_detection(_three_leaks(), [], max_distance=0.1)
    key, ids = first.new_clusters[0]
    existing = [ExistingCluster("k1", key, "후보", frozenset(ids))]
    again = plan_detection(_three_leaks(), existing, max_distance=0.1)
    assert again.new_clusters == [] and again.additions == []


def test_decided_cluster_reports_are_not_counted_again_with_similarity() -> None:
    key = plan_detection(_three_leaks(), [], max_distance=0.1).new_clusters[0][0]
    rejected = ExistingCluster("k1", key, "기각", frozenset({"a", "b", "c"}))
    plan = plan_detection(_three_leaks(), [rejected], max_distance=0.1)
    assert plan.new_clusters == [] and plan.additions == []


# ---------------------------------------------------------------- 설정·변환


def test_similarity_distance_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DETECTION_SIMILARITY_DISTANCE", raising=False)
    assert similarity_distance() == DEFAULT_SIMILARITY_DISTANCE
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0.4")
    assert similarity_distance() == 0.4
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0")
    assert similarity_distance() == 0.0
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "abc")
    assert similarity_distance() == DEFAULT_SIMILARITY_DISTANCE


def test_vector_text_round_trip() -> None:
    assert parse_vector(None) is None
    assert parse_vector("[]") is None
    values = (0.25, -0.5, 1.0)
    assert parse_vector(format_vector(values)) == values


# ---------------------------------------------------------------- 임베딩 채우기


class _Result:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeDB:
    """SELECT는 준비된 행을, UPDATE는 기록만 한다."""

    def __init__(self, pending: list[tuple[str, str]]) -> None:
        self.pending = pending
        self.updates: list[dict[str, Any]] = []

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> _Result:
        sql = str(stmt)
        if sql.startswith("UPDATE reports"):
            assert params is not None
            self.updates.append(params)
            return _Result([])
        assert "embedding IS NULL" in sql
        return _Result(list(self.pending))


def _fake_embed(texts: Sequence[str]) -> list[list[float]]:
    return [[float(len(t)), 0.0] for t in texts]


def test_embed_missing_reports_writes_vectors() -> None:
    db = _FakeDB([("r1", "화장실 물이 샘"), ("r2", "복도 불 나감")])
    assert embed_missing_reports(db, _fake_embed) == (2, False)
    assert [u["id"] for u in db.updates] == ["r1", "r2"]
    assert db.updates[0]["v"] == format_vector([float(len("화장실 물이 샘")), 0.0])


def test_embed_missing_reports_nothing_pending() -> None:
    db = _FakeDB([])
    assert embed_missing_reports(db, _fake_embed) == (0, False)
    assert db.updates == []


def test_embedding_failure_does_not_touch_db() -> None:
    def broken(texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("Gemini down")

    db = _FakeDB([("r1", "화장실 물이 샘")])
    assert embed_missing_reports(db, broken) == (0, True)
    assert db.updates == []


def test_embedding_count_mismatch_is_treated_as_failure() -> None:
    db = _FakeDB([("r1", "a"), ("r2", "b")])
    assert embed_missing_reports(db, lambda texts: [[1.0, 0.0]]) == (0, True)
    assert db.updates == []


# ---------------------------------------------------------------- run_detection 흐름


def _patch_db_layer(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    monkeypatch.setattr(detection, "load_thresholds", lambda db: (3, 72))
    monkeypatch.setattr(detection, "load_points", lambda db: _three_leaks())
    monkeypatch.setattr(detection, "load_existing", lambda db: [])
    monkeypatch.setattr(detection, "apply_plan", lambda db, plan: calls.append("apply"))
    monkeypatch.setattr(detection, "load_unembedded", lambda db, limit=200: [("x", "y")])

    def fake_embed(db: Any) -> tuple[int, bool]:
        calls.append("embed")
        return 2, False

    monkeypatch.setattr(detection, "embed_missing_reports", fake_embed)


def test_run_detection_embeds_then_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0.1")
    calls: list[str] = []
    _patch_db_layer(monkeypatch, calls)
    result = run_detection(object())
    assert calls == ["embed", "apply"]
    assert result.reports_embedded == 2 and not result.embedding_error
    assert result.clusters_created == 1 and result.merged_groups == 1
    assert result.similarity_distance == 0.1


def test_run_detection_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0.1")
    calls: list[str] = []
    _patch_db_layer(monkeypatch, calls)
    result = run_detection(object(), dry_run=True)
    assert calls == []
    assert result.embedding_pending == 1 and result.reports_embedded == 0
    assert result.clusters_created == 1 and result.dry_run


def test_run_detection_with_similarity_off_uses_exact_detail_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DETECTION_SIMILARITY_DISTANCE", "0")
    calls: list[str] = []
    _patch_db_layer(monkeypatch, calls)
    result = run_detection(object())
    assert calls == ["apply"]  # 임베딩도 안 채움
    assert result.clusters_created == 0 and result.similarity_distance == 0.0
