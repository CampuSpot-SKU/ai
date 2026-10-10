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

임베딩 유사도 보조 (1-8b, 명세 3-3)
- 세부위치 표기가 달라도("3동 2층 화장실" / "3동 남자화장실 물샘") 같은 건물 + 같은 카테고리 안에서 신고 설명의
  임베딩(`reports.embedding`, 코사인 거리)이 `similarity_distance()` 이하면 같은 곳으로 본다(연쇄도 허용).
- 임베딩은 접수 경로가 아니라 이 배치가 아직 비어 있는 신고만 채운다(접수가 느려지거나 Gemini 장애로 접수가
  실패할 일이 없음). 임베딩에 실패해도 탐지는 멈추지 않고 표기가 같은 신고끼리만 묶는다.
- 묶음의 대표 세부위치는 가장 많이 쓰인 표기(같으면 먼저 접수된 쪽)다.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_COUNT = 3
DEFAULT_THRESHOLD_HOURS = 72

# 같은 건물·카테고리에서 설명 임베딩의 평균 코사인 거리가 이 값 이하면 같은 곳으로 본다.
# 10/10 실데이터: 같은 내용 0~0.004, 짧은 고장 문장끼리는 서로 달라도 0.09~0.25라 보수적으로 0.05. 거짓 병합이 놓침보다 해롭다.
# S-2 데모 데이터로 다시 확인할 값 —
# DETECTION_SIMILARITY_DISTANCE로 덮어쓴다(0 이하면 임베딩 보조를 끈다).
DEFAULT_SIMILARITY_DISTANCE = 0.05
EMBED_LIMIT = 200  # 배치 한 번에 임베딩을 채울 신고 수 상한
EMBED_TASK = "SEMANTIC_SIMILARITY"

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
    embedding: tuple[float, ...] | None = None  # 아직 없으면 None — 표기가 같은 신고끼리만 묶임

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
    groups: int = 0  # 같은 곳으로 본 묶음 수
    merged_groups: int = 0  # 그중 세부위치 표기가 둘 이상 섞인(= 임베딩이 이어 붙인) 묶음 수


@dataclass
class DetectionResult:
    reports_scanned: int = 0
    groups: int = 0
    merged_groups: int = 0
    clusters_created: int = 0
    clusters_extended: int = 0
    links_added: int = 0
    reports_embedded: int = 0
    embedding_pending: int = 0  # dry_run일 때 채우지 않고 남겨 둔 신고 수
    embedding_error: bool = False
    similarity_distance: float = 0.0
    dry_run: bool = False
    explain: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reports_scanned": self.reports_scanned,
            "groups": self.groups,
            "merged_groups": self.merged_groups,
            "clusters_created": self.clusters_created,
            "clusters_extended": self.clusters_extended,
            "links_added": self.links_added,
            "reports_embedded": self.reports_embedded,
            "embedding_pending": self.embedding_pending,
            "embedding_error": self.embedding_error,
            "similarity_distance": self.similarity_distance,
            "dry_run": self.dry_run,
            **({"explain": self.explain} if self.explain is not None else {}),
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


def similarity_distance() -> float:
    """임베딩 보조의 코사인 거리 기준. 환경변수 DETECTION_SIMILARITY_DISTANCE로 조정, 0 이하면 끔."""
    raw = os.environ.get("DETECTION_SIMILARITY_DISTANCE", "").strip()
    if not raw:
        return DEFAULT_SIMILARITY_DISTANCE
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_SIMILARITY_DISTANCE


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """코사인 거리(1 - 코사인 유사도). 0이면 같은 방향, 영벡터는 가장 먼 값(1.0)으로 본다."""
    if len(a) != len(b):
        raise ValueError("임베딩 길이가 다릅니다.")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _components(size: int, linked: Callable[[int, int], bool]) -> list[list[int]]:
    """0..size-1 번 항목을 `linked(x, y)`로 이어진 것끼리 묶는다(연쇄 포함). 각 묶음은 번호 오름차순."""
    parent = list(range(size))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x in range(size):
        for y in range(x + 1, size):
            if linked(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
    members: dict[int, list[int]] = defaultdict(list)
    for x in range(size):
        members[find(x)].append(x)
    return list(members.values())


def _average_distance(a: Sequence[ReportPoint], b: Sequence[ReportPoint]) -> float | None:
    """두 묶음 사이 평균 코사인 거리. 양쪽 모두 임베딩이 있는 쌍만 센다(하나도 없으면 None)."""
    total, n = 0.0, 0
    for x in a:
        if x.embedding is None:
            continue
        for y in b:
            if y.embedding is None:
                continue
            total += cosine_distance(x.embedding, y.embedding)
            n += 1
    return total / n if n else None


def _merge_by_average(clusters: list[list[ReportPoint]], limit: float) -> list[list[ReportPoint]]:
    """평균 연결: 평균 거리가 가장 가까운 두 묶음부터 `limit` 이하일 때까지 합친다.
    다리 역할을 하는 문장 하나로 서로 다른 묶음이 이어 붙는 것(연쇄)을 막는다."""
    groups = [list(c) for c in clusters]
    while len(groups) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = _average_distance(groups[i], groups[j])
                if d is not None and d <= limit and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return groups


def group_points(
    points: Sequence[ReportPoint], max_distance: float | None = None
) -> list[list[ReportPoint]]:
    """같은 곳으로 볼 신고끼리 묶는다. 건물·카테고리가 같은 신고 안에서 세부위치 표기가 같으면 먼저 한 묶음,
    (`max_distance`가 있으면) 그 묶음끼리 임베딩 평균 거리가 `max_distance` 이하일 때 합친다(평균 연결)."""
    limit = max_distance if max_distance is not None and max_distance > 0 else None
    buckets: dict[tuple[str, str], list[ReportPoint]] = defaultdict(list)
    for p in points:
        buckets[(p.building_id, p.category_id)].append(p)

    groups: list[list[ReportPoint]] = []
    for bucket_key in sorted(buckets):
        bucket = sorted(buckets[bucket_key], key=lambda p: (p.created_at, p.id))

        def same_detail(x: int, y: int, b: list[ReportPoint] = bucket) -> bool:
            return b[x].detail == b[y].detail

        same = _components(len(bucket), same_detail)
        base = [[bucket[x] for x in idx] for idx in same]
        groups.extend(_merge_by_average(base, limit) if limit is not None else base)
    for g in groups:
        g.sort(key=lambda p: (p.created_at, p.id))
    groups.sort(key=group_key)
    return groups


def explain_pairs(points: Sequence[ReportPoint], top: int = 40) -> list[dict[str, Any]]:
    """기준값을 고를 때 보는 자료: 건물·카테고리가 같고 세부위치 표기가 다른 신고 쌍을 거리 가까운 순으로."""
    buckets: dict[tuple[str, str], list[ReportPoint]] = defaultdict(list)
    for p in points:
        if p.embedding is not None:
            buckets[(p.building_id, p.category_id)].append(p)
    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        for i, a in enumerate(bucket):
            for b in bucket[i + 1 :]:
                if a.detail == b.detail or a.embedding is None or b.embedding is None:
                    continue
                rows.append(
                    {
                        "distance": round(cosine_distance(a.embedding, b.embedding), 4),
                        "a": a.id,
                        "a_detail": a.detail,
                        "b": b.id,
                        "b_detail": b.detail,
                    }
                )
    rows.sort(key=lambda r: r["distance"])
    return rows[:top]


def group_key(group: Sequence[ReportPoint]) -> GroupKey:
    """묶음의 (건물, 대표 세부위치, 카테고리). 대표 세부위치는 가장 많이 쓰인 표기, 같으면 먼저 접수된 쪽."""
    counts = Counter(p.detail for p in group)
    first_seen: dict[str, datetime] = {}
    for p in sorted(group, key=lambda p: (p.created_at, p.id)):
        first_seen.setdefault(p.detail, p.created_at)
    detail = min(counts, key=lambda d: (-counts[d], first_seen[d], d))
    return (group[0].building_id, detail, group[0].category_id)


def plan_detection(
    points: Sequence[ReportPoint],
    existing: Sequence[ExistingCluster],
    count: int = DEFAULT_THRESHOLD_COUNT,
    hours: int = DEFAULT_THRESHOLD_HOURS,
    max_distance: float | None = None,
) -> DetectionPlan:
    """신고와 기존 후보를 보고 DB에 반영할 일을 계산한다 (DB 접근 없음).

    `max_distance`가 없으면 세부위치 표기가 같은 신고끼리만 묶는다(1-8 동작 그대로)."""
    groups = group_points(points, max_distance)

    clusters_by_place: dict[tuple[str, str], list[ExistingCluster]] = defaultdict(list)
    for c in existing:
        clusters_by_place[(c.key[0], c.key[2])].append(c)

    plan = DetectionPlan(groups=len(groups))
    for group in groups:
        key = group_key(group)
        group_ids = {p.id for p in group}
        details = {p.detail for p in group}
        if len(details) > 1:
            plan.merged_groups += 1
        decided: set[str] = set()  # 이미 승격·기각된 후보에 들어간 신고 — 다시 세지 않음
        open_cluster: ExistingCluster | None = None
        for c in clusters_by_place.get((key[0], key[2]), []):
            # 같은 세부위치 표기이거나 이미 같은 신고를 담고 있는 후보가 이 묶음의 후보다
            if c.key[1] not in details and not (c.report_ids & group_ids):
                continue
            if c.status == STATUS_CANDIDATE:
                if open_cluster is None:
                    open_cluster = c
            else:
                decided |= c.report_ids
        fresh = [p for p in group if p.id not in decided]
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
            "SELECT id, building_id, btrim(detail), category_id, created_at, CAST(embedding AS text) "
            "FROM reports "
            "WHERE building_id IS NOT NULL AND detail IS NOT NULL AND btrim(detail) <> '' "
            "ORDER BY created_at, id"
        )
    ).all()
    return [
        ReportPoint(str(r[0]), str(r[1]), str(r[2]), str(r[3]), r[4], parse_vector(r[5]))
        for r in rows
    ]


def parse_vector(raw: Any) -> tuple[float, ...] | None:
    """pgvector 글자 표현 `[0.1,0.2,...]` → 숫자 튜플. 비어 있으면 None."""
    if raw is None:
        return None
    values = json.loads(str(raw))
    return tuple(float(v) for v in values) if values else None


def format_vector(values: Sequence[float]) -> str:
    """숫자 목록 → pgvector 글자 표현 `[0.1,0.2,...]` (CAST(:v AS vector)에 넣는다)."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


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


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    from ai import rag  # Gemini·DB 라이브러리는 실제로 쓸 때만 불러온다

    return rag.embed_texts(texts, task=EMBED_TASK)


def load_unembedded(db: Any, limit: int = EMBED_LIMIT) -> list[tuple[str, str]]:
    """임베딩이 아직 없는 신고 (id, 설명) — 오래된 것부터. 설명이 비면 건너뛴다."""
    from sqlalchemy import text

    rows = db.execute(
        text(
            "SELECT id, description FROM reports "
            "WHERE embedding IS NULL AND btrim(description) <> '' "
            "ORDER BY created_at, id LIMIT :n"
        ),
        {"n": limit},
    ).all()
    return [(str(r[0]), str(r[1])) for r in rows]


def embed_missing_reports(
    db: Any,
    embed: Callable[[Sequence[str]], list[list[float]]] | None = None,
    limit: int = EMBED_LIMIT,
) -> tuple[int, bool]:
    """비어 있는 신고 임베딩을 채운다. (채운 수, 임베딩 실패 여부)를 돌려준다.

    임베딩 호출이 실패하면 DB는 건드리지 않고 (0, True) — 탐지는 표기가 같은 신고끼리만 묶고 계속된다.
    DB 쓰기 오류는 그대로 올려 보낸다(트랜잭션을 망가뜨린 채 이어가지 않도록). 커밋은 호출한 쪽이 한다."""
    from sqlalchemy import text

    pending = load_unembedded(db, limit)
    if not pending:
        return 0, False
    try:
        vectors = (embed or _default_embed)([desc for _, desc in pending])
        if len(vectors) != len(pending):
            raise RuntimeError("임베딩 개수가 신고 수와 다릅니다.")
    except Exception:
        logger.exception("신고 임베딩 생성 실패 — 이번 탐지는 표기가 같은 신고끼리만 묶습니다")
        return 0, True
    update = text("UPDATE reports SET embedding = CAST(:v AS vector) WHERE id = :id")
    for (report_id, _), vec in zip(pending, vectors, strict=True):
        db.execute(update, {"v": format_vector(vec), "id": report_id})
    return len(pending), False


def run_detection(db: Any, dry_run: bool = False, explain: bool = False) -> DetectionResult:
    """탐지 배치 한 번. `dry_run`이면 계산만 하고 DB에는 쓰지 않는다. 커밋은 호출한 쪽이 한다.

    순서: (실행이면) 비어 있는 신고 임베딩 채우기 → 신고 읽기 → 묶기·계획 → DB 반영."""
    count, hours = load_thresholds(db)
    max_distance = similarity_distance()
    use_embedding = max_distance > 0
    embedded, embed_error, pending = 0, False, 0
    if use_embedding:
        if dry_run:
            pending = len(load_unembedded(db))
        else:
            embedded, embed_error = embed_missing_reports(db)
    points = load_points(db)
    plan = plan_detection(
        points, load_existing(db), count, hours, max_distance if use_embedding else None
    )
    result = DetectionResult(
        reports_scanned=len(points),
        groups=plan.groups,
        merged_groups=plan.merged_groups,
        clusters_created=len(plan.new_clusters),
        clusters_extended=len(plan.additions),
        links_added=sum(len(ids) for _, ids in plan.new_clusters)
        + sum(len(ids) for _, ids in plan.additions),
        reports_embedded=embedded,
        embedding_pending=pending,
        embedding_error=embed_error,
        similarity_distance=max_distance if use_embedding else 0.0,
        dry_run=dry_run,
        explain=explain_pairs(points) if explain else None,
    )
    if not dry_run:
        apply_plan(db, plan)
    return result
