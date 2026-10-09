"""예측 계산 — 같은 곳에서 얼마 간격으로 신고가 반복되는지 구해 다음 예상 시점을 낸다 (1-11, 명세 3-3).

규칙 (명세 3-3 그대로, 복잡한 모델 없이 평균 간격)
- 같은 건물 + 카테고리 조합의 과거 신고가 3건 이상이어야 계산한다(미달이면 행을 만들지 않음 →
  관리자 화면의 "데이터 부족, 예측 불가"는 API가 계산된 행만 내려주는 것으로 처리된다).
- 평균 재발 주기(일) = 신고 시점 사이 간격의 평균, 다음 예상 시점 = 마지막 신고 시점 + 평균 주기.
- 결과 표 `prediction_stats`는 매번 통째로 다시 만든다(파생 데이터라 지우고 다시 써도 안전 — 같은
  배치를 여러 번 돌려도 같은 결과). `detail`은 비워 둔다(명세가 건물 + 카테고리 기준).
- DB 라이브러리는 함수 안에서만 쓴다(없는 환경에서도 계산 테스트가 돌도록).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Any

MIN_OCCURRENCES = 3
SECONDS_PER_DAY = 86400.0

# (건물 id, 카테고리 id) — 문자열
PredictionKey = tuple[str, str]


@dataclass(frozen=True)
class PredictionStat:
    building_id: str
    category_id: str
    avg_recurrence_days: float
    last_occurred_at: datetime
    predicted_next_at: datetime


def compute_stat(times: Sequence[datetime]) -> tuple[float, datetime, datetime] | None:
    """신고 시각들로 (평균 주기 일, 마지막 발생, 다음 예상 시점)을 계산. 3건 미만이면 None."""
    if len(times) < MIN_OCCURRENCES:
        return None
    ordered = sorted(times)
    gaps = [(b - a).total_seconds() / SECONDS_PER_DAY for a, b in pairwise(ordered)]
    avg = sum(gaps) / len(gaps)
    last = ordered[-1]
    return avg, last, last + timedelta(days=avg)


def build_stats(rows: Sequence[tuple[str, str, datetime]]) -> list[PredictionStat]:
    """(건물 id, 카테고리 id, 신고 시각) 목록 → 예측 가능한 조합의 통계."""
    grouped: dict[PredictionKey, list[datetime]] = defaultdict(list)
    for building_id, category_id, created_at in rows:
        grouped[(building_id, category_id)].append(created_at)
    stats: list[PredictionStat] = []
    for (building_id, category_id), times in sorted(grouped.items()):
        computed = compute_stat(times)
        if computed is None:
            continue
        avg, last, nxt = computed
        stats.append(PredictionStat(building_id, category_id, avg, last, nxt))
    return stats


# ---------------------------------------------------------------- DB 읽기·쓰기


def load_rows(db: Any) -> list[tuple[str, str, datetime]]:
    from sqlalchemy import text

    rows = db.execute(
        text(
            "SELECT building_id, category_id, created_at FROM reports "
            "WHERE building_id IS NOT NULL ORDER BY created_at, id"
        )
    ).all()
    return [(str(r[0]), str(r[1]), r[2]) for r in rows]


def replace_stats(db: Any, stats: Sequence[PredictionStat]) -> None:
    """`prediction_stats`를 계산 결과로 통째로 바꾼다. 커밋은 호출한 쪽이 한다."""
    from sqlalchemy import text

    db.execute(text("DELETE FROM prediction_stats"))
    insert = text(
        "INSERT INTO prediction_stats "
        "(building_id, category_id, avg_recurrence_days, last_occurred_at, predicted_next_at) "
        "VALUES (:b, :c, :avg, :last, :next)"
    )
    for s in stats:
        db.execute(
            insert,
            {
                "b": s.building_id,
                "c": s.category_id,
                "avg": s.avg_recurrence_days,
                "last": s.last_occurred_at,
                "next": s.predicted_next_at,
            },
        )


def run_prediction(db: Any, dry_run: bool = False) -> dict[str, Any]:
    """예측 배치 한 번. `dry_run`이면 계산만 하고 DB에는 쓰지 않는다."""
    rows = load_rows(db)
    stats = build_stats(rows)
    if not dry_run:
        replace_stats(db, stats)
    return {
        "reports_scanned": len(rows),
        "predictions": len(stats),
        "dry_run": dry_run,
    }
