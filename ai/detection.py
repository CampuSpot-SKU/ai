"""탐지 배치 — 같은 곳에서 반복되는 신고를 "문제 후보"로 묶는다 (1-8, 명세 3-3).

흐름: 신고 읽기(load_points) → 묶음 찾기(find_bursts) → 기존 후보와 맞춰 계획 세우기(plan_detection)
      → DB 반영(apply_plan). `run_detection`이 이 순서를 한 번에 실행한다.

규칙 (명세 3-3, 임계치는 설정 `detection`의 threshold_count / threshold_hours — 기본 3건 / 72시간)
- 같은 건물 + 세부위치 + 카테고리의 신고가 threshold_hours 시간 안에 threshold_count건 이상이면 후보.
  명세는 "건물 + 세부위치"인데 `problem_clusters.category_id`가 필수라 카테고리까지 같이 묶는다.
- 건물이나 세부위치가 비어 있는 신고는 "같은 곳"을 말할 수 없어 탐지에서 뺀다.
- 겹치는 시간창은 하나의 묶음으로 합친다(3건씩 두 창이 겹치면 후보 1개).
- 매일 돌려도 중복이 생기지 않는다: 같은 곳의 `후보`가 이미 있으면 새 신고만 거기에 더하고,
  이미 `승격`·`기각`한 후보에 들어간 신고는 다시 세지 않는다(관리자가 결정한 것을 되살리지 않음).
  같은 곳에 열려 있는 후보는 최대 1개다(시간이 떨어진 묶음이 여러 개여도 한 후보에 모은다).
- DB 라이브러리는 함수 안에서만 쓴다(없는 환경에서도 묶기 로직 테스트가 돌도록).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

DEFAULT_THRESHOLD_COUNT = 3
DEFAULT_THRESHOLD_HOURS = 72

STATUS_CANDIDATE = "후보"

# (건물 id, 세부위치, 카테고리 id) — 모두 문자열
GroupKey = tuple[str, str, str]


@dataclass(frozen=True)
class ReportPoint:
    """탐지에 쓰는 신고 한 건."""

    id: str
    building_id: str
    detail: str
    category_id: str
    created_at: datetime

    @property
    def key(self) -> GroupKey:
        return (self.building_id, self.detail, self.category_id)


@dataclass(frozen=True)
class ExistingCluster:
    """DB에 이미 있는 후보/승격/기각 한 건과 거기에 묶인 신고 id."""

    id: str
    key: GroupKey
    status: str
    report_ids: frozenset[str]


@dataclass
class DetectionPlan:
    """DB에 반영할 일. 새 후보 만들기 / 열려 있는 후보에 신고 더하기."""

    new_clusters: list[tuple[GroupKey, list[str]]] = field(default_factory=list)
    additions: list[tuple[str, list[str]]] = field(default_factory=list)  # (cluster_id, 신고 id들)


@dataclass
class DetectionResult:
    reports_scanned: int = 0
    groups: int = 0
    clusters_created: int = 0
    clusters_extended: int = 0
    links_added: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "reports_scanned": self.reports_scanned,
            "groups": self.groups,
            "clusters_created": self.clusters_created,
            "clusters_extended": self.clusters_extended,
            "links_added": self.links_added,
            "dry_run": self.dry_run,
        }


# ---------------------------------------------------------------- 묶음 찾기 (순수 로직)


def find_bursts(items: Sequence[tuple[datetime, str]], count: int, hours: int) -> list[list[str]]:
    """(시각, 신고 id) 목록에서 `hours`시간 안에 `count`건 이상 몰린 묶음을 찾는다.

    - 시간창 경계는 포함(정확히 `hours`시간 차이도 같은 창).
    - 겹치는 창은 하나로 합쳐 신고 id를 시각순으로 돌려준다.
    """
    if count < 1:
        raise ValueError("count는 1 이상이어야 합니다.")
    ordered = sorted(items, key=lambda x: (x[0], x[1]))
    window = timedelta(hours=hours)
    bursts: list[list[str]] = []
    left = 0
    cur_start = -1  # 지금 합치는 묶음의 인덱스 범위 [cur_start, cur_end]
    cur_end = -1
    for right, (t_right, _) in enumerate(ordered):
        while t_right - ordered[left][0] > window:
            left += 1
        if right - left + 1 < count:
            continue
        if cur_start >= 0 and left <= cur_end:
            cur_end = right  # 이어지는 창 → 같은 묶음
        else:
            if cur_start >= 0:
                bursts.append([i for _, i in ordered[cur_start : cur_end + 1]])
            cur_start, cur_end = left, right
    if cur_start >= 0:
        bursts.append([i for _, i in ordered[cur_start : cur_end + 1]])
    return bursts


def plan_detection(
    points: Sequence[ReportPoint],
    existing: Sequence[ExistingCluster],
    count: int = DEFAULT_THRESHOLD_COUNT,
    hours: int = DEFAULT_THRESHOLD_HOURS,
) -> DetectionPlan:
    """신고와 기존 후보를 보고 DB에 반영할 일을 계산한다 (DB 접근 없음)."""
    by_key: dict[GroupKey, list[ReportPoint]] = defaultdict(list)
    for p in points:
        by_key[p.key].append(p)

    clusters_by_key: dict[GroupKey, list[ExistingCluster]] = defaultdict(list)
    for c in existing:
        clusters_by_key[c.key].append(c)

    plan = DetectionPlan()
    for key in sorted(by_key):
        decided: set[str] = set()  # 이미 승격·기각된 후보에 들어간 신고 — 다시 세지 않음
        open_cluster: ExistingCluster | None = None
        for c in clusters_by_key.get(key, []):
            if c.status == STATUS_CANDIDATE:
                if open_cluster is None:
                    open_cluster = c
            else:
                decided |= c.report_ids
        fresh = [p for p in by_key[key] if p.id not in decided]
        bursts = find_bursts([(p.created_at, p.id) for p in fresh], count, hours)
        if not bursts:
            continue
        have = set(open_cluster.report_ids) if open_cluster else set()
        picked: list[str] = []
        for burst in bursts:
            for rid in burst:
                if rid not in have:
                    have.add(rid)
                    picked.append(rid)
        if not picked:
            continue
        if open_cluster is not None:
            plan.additions.append((open_cluster.id, picked))
        else:
            plan.new_clusters.append((key, picked))
    return plan


# ---------------------------------------------------------------- DB 읽기·쓰기


def load_thresholds(db: Any) -> tuple[int, int]:
    """설정 `detection`(detection_config)의 임계치. 행이 없으면 명세 기본값."""
    from sqlalchemy import text

    row = db.execute(
        text("SELECT threshold_count, threshold_hours FROM detection_config ORDER BY id LIMIT 1")
    ).first()
    if row is None:
        return DEFAULT_THRESHOLD_COUNT, DEFAULT_THRESHOLD_HOURS
    return int(row[0]), int(row[1])


def load_points(db: Any) -> list[ReportPoint]:
    """건물과 세부위치가 모두 있는 신고만 읽는다."""
    from sqlalchemy import text

    rows = db.execute(
        text(
            "SELECT id, building_id, btrim(detail), category_id, created_at FROM reports "
            "WHERE building_id IS NOT NULL AND detail IS NOT NULL AND btrim(detail) <> '' "
            "ORDER BY created_at, id"
        )
    ).all()
    return [ReportPoint(str(r[0]), str(r[1]), str(r[2]), str(r[3]), r[4]) for r in rows]


def load_existing(db: Any) -> list[ExistingCluster]:
    """이미 있는 후보·승격·기각과 묶인 신고. 건물이나 세부위치가 없는 후보는 비교 대상이 아니다."""
    from sqlalchemy import text

    clusters = db.execute(
        text(
            "SELECT id, building_id, btrim(detail), category_id, status::text "
            "FROM problem_clusters WHERE building_id IS NOT NULL AND detail IS NOT NULL "
            "ORDER BY detected_at, id"
        )
    ).all()
    members: dict[str, set[str]] = defaultdict(set)
    for cluster_id, report_id in db.execute(
        text("SELECT cluster_id, report_id FROM problem_cluster_reports")
    ).all():
        members[str(cluster_id)].add(str(report_id))
    return [
        ExistingCluster(
            id=str(c[0]),
            key=(str(c[1]), str(c[2]), str(c[3])),
            status=str(c[4]),
            report_ids=frozenset(members.get(str(c[0]), set())),
        )
        for c in clusters
    ]


def apply_plan(db: Any, plan: DetectionPlan) -> None:
    """계획을 DB에 쓴다. 커밋은 호출한 쪽이 한다."""
    from sqlalchemy import text

    link = text(
        "INSERT INTO problem_cluster_reports (cluster_id, report_id) VALUES (:c, :r) "
        "ON CONFLICT DO NOTHING"
    )
    for (building_id, detail, category_id), report_ids in plan.new_clusters:
        cluster_id = db.execute(
            text(
                "INSERT INTO problem_clusters (building_id, detail, category_id) "
                "VALUES (:b, :d, :c) RETURNING id"
            ),
            {"b": building_id, "d": detail, "c": category_id},
        ).scalar_one()
        for rid in report_ids:
            db.execute(link, {"c": str(cluster_id), "r": rid})
    for cluster_id, report_ids in plan.additions:
        for rid in report_ids:
            db.execute(link, {"c": cluster_id, "r": rid})


def run_detection(db: Any, dry_run: bool = False) -> DetectionResult:
    """탐지 배치 한 번. `dry_run`이면 계산만 하고 DB에는 쓰지 않는다. 커밋은 호출한 쪽이 한다."""
    count, hours = load_thresholds(db)
    points = load_points(db)
    plan = plan_detection(points, load_existing(db), count, hours)
    result = DetectionResult(
        reports_scanned=len(points),
        groups=len({p.key for p in points}),
        clusters_created=len(plan.new_clusters),
        clusters_extended=len(plan.additions),
        links_added=sum(len(ids) for _, ids in plan.new_clusters)
        + sum(len(ids) for _, ids in plan.additions),
        dry_run=dry_run,
    )
    if not dry_run:
        apply_plan(db, plan)
    return result
