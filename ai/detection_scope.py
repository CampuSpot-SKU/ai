"""건물 전체 탐지 — 같은 건물에서 호실이 달라도 하나의 원인(건물 전체 정전·네트워크 장애 등)이면 한 묶음으로 본다.

기존 탐지(detection.py)는 같은 건물 + 같은 세부위치 + 같은 카테고리만 같은 곳으로 본다. 여기서는 그 앞 단계로,
같은 건물 + 같은 카테고리에서 짧은 시간에 여러 세부위치의 신고가 몰렸을 때 Gemini에게 "건물 전체 원인인가, 호실·층마다
따로 생긴 일인가"를 묻는다. 건물 전체라고 한 신고만 세부위치 "건물 전체"인 묶음으로 만들고, 나머지는 기존 방식이 그대로 처리한다.

안전장치 (이 배치는 접수 경로와 무관 — 실패해도 신고 접수·기존 탐지에 영향 없음)
- Gemini 호출·응답 검사에 실패하면 그 묶음은 건너뛴다(기존 탐지만 동작). DETECTION_SCOPE_AI=0이면 아예 끈다.
- 확신이 없으면 false가 기본이고, 건물 전체로 인정하려면 신고가 임계 건수 이상이고 세부위치가 둘 이상이어야 한다.
- 한 번 실행에 부르는 횟수 상한(MAX_CALLS)이 있다.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ai.detection import (
    STATUS_CANDIDATE,
    WHOLE_BUILDING,
    ExistingCluster,
    GroupKey,
    ReportPoint,
    find_bursts,
)

logger = logging.getLogger(__name__)

MAX_CALLS = 8  # 한 번 실행에서 Gemini를 부르는 최대 횟수
MAX_CASES = 12  # 한 번에 보여주는 신고 수 상한
MAX_REASON_LEN = 80
DEFAULT_MODEL = "gemini-3.5-flash-lite"
_PROMPT_PATH = Path(__file__).parent / "prompts" / "detection_scope.md"


class ScopeVerdict(BaseModel):
    """Gemini가 출력해야 하는 JSON 형식 (response_schema로 강제)."""

    building_wide: bool
    report_numbers: list[int] = Field(default_factory=list)
    reason: str = ""


# (건물 이름, 카테고리 이름, 신고들[시간순]) → 판단. 테스트에서는 가짜로 바꿔 끼운다.
ScopeJudge = Callable[[str, str, Sequence[ReportPoint]], ScopeVerdict]


@dataclass
class BuildingWidePlan:
    new_clusters: list[tuple[GroupKey, list[str]]] = field(default_factory=list)
    additions: list[tuple[str, list[str]]] = field(default_factory=list)  # (cluster_id, 신고 id들)
    calls: int = 0
    reasons: list[str] = field(default_factory=list)
    errors: int = 0  # Gemini 호출이 실패한 횟수
    notes: list[str] = field(default_factory=list)  # 묶음마다 AI 판단 한 줄 (운영 확인용)

    def consumed_ids(self) -> set[str]:
        ids: set[str] = set()
        for _, rids in self.new_clusters:
            ids.update(rids)
        for _, rids in self.additions:
            ids.update(rids)
        return ids


def scope_enabled() -> bool:
    return os.environ.get("DETECTION_SCOPE_AI", "1").strip() not in ("0", "false", "off")


def find_building_wide(
    points: Sequence[ReportPoint],
    existing: Sequence[ExistingCluster],
    count: int,
    hours: int,
    judge: ScopeJudge,
    names: dict[str, str],
    max_calls: int = MAX_CALLS,
) -> BuildingWidePlan:
    """건물 전체 묶음 계획 (DB 접근 없음). `names`는 건물·카테고리 id → 이름."""
    whole = [c for c in existing if c.key[1] == WHOLE_BUILDING]
    taken: set[str] = set()  # 이미 건물 전체 묶음(후보·승격·기각)에 들어간 신고는 다시 세지 않는다
    for c in whole:
        taken |= c.report_ids

    buckets: dict[tuple[str, str], list[ReportPoint]] = defaultdict(list)
    for p in points:
        if p.id not in taken:
            buckets[(p.building_id, p.category_id)].append(p)

    plan = BuildingWidePlan()
    for (building_id, category_id), pts in sorted(buckets.items()):
        pts.sort(key=lambda p: (p.created_at, p.id))
        open_whole = next(
            (c for c in whole if c.key[0] == building_id and c.key[2] == category_id and c.status == STATUS_CANDIDATE),
            None,
        )
        by_id = {p.id: p for p in pts}
        for burst in find_bursts([(p.created_at, p.id) for p in pts], count, hours):
            members = [by_id[i] for i in burst]
            if len({p.detail for p in members}) < 2:
                continue  # 세부위치가 하나뿐이면 기존 탐지가 처리
            if plan.calls >= max_calls:
                return plan
            plan.calls += 1
            shown = members[-MAX_CASES:]
            try:
                verdict = judge(names.get(building_id, "이 건물"), names.get(category_id, ""), shown)
            except Exception:  # Gemini·네트워크 오류 전부: 이 묶음만 건너뛴다
                logger.exception("건물 전체 판단 실패 — 건너뜀")
                plan.errors += 1
                plan.notes.append(f"{names.get(building_id, '?')}/{names.get(category_id, '?')}: 호출 실패")
                continue
            plan.notes.append(
                f"{names.get(building_id, '?')}/{names.get(category_id, '?')} "
                f"{'건물 전체' if verdict.building_wide else '개별'}({len(shown)}건): "
                + " ".join(verdict.reason.split())[:MAX_REASON_LEN]
            )
            picked = sorted({n for n in verdict.report_numbers if 1 <= n <= len(shown)})
            ids = [shown[n - 1].id for n in picked]
            chosen = [shown[n - 1] for n in picked]
            if not verdict.building_wide or len(ids) < count or len({p.detail for p in chosen}) < 2:
                continue
            plan.reasons.append(" ".join(verdict.reason.split())[:MAX_REASON_LEN])
            if open_whole is not None:
                plan.additions.append((open_whole.id, ids))
            else:
                plan.new_clusters.append(((building_id, WHOLE_BUILDING, category_id), ids))
                open_whole = None
    return plan


def _prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def gemini_scope_judge(building: str, category: str, cases: Sequence[ReportPoint]) -> ScopeVerdict:
    """Gemini로 건물 전체 여부를 묻는다. 1회 재시도, 그래도 실패하면 예외."""
    from google.genai import types

    from ai.intent import _client

    lines = [f"## 건물: {building}", f"## 카테고리: {category}", "\n## 신고들 (접수 순)"]
    for n, p in enumerate(cases, start=1):
        lines.append(f"{n}. [{p.created_at:%m/%d %H:%M}] 위치: {p.detail} / 내용: {p.description or '(없음)'}")
    content = "\n".join(lines)
    last: Exception | None = None
    for _ in (1, 2):
        try:
            response = _client().models.generate_content(
                model=os.environ.get("GEMINI_SCOPE_MODEL") or os.environ.get("GEMINI_INTENT_MODEL") or DEFAULT_MODEL,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=_prompt(),
                    response_mime_type="application/json",
                    response_schema=ScopeVerdict,
                    temperature=0,
                ),
            )
            return ScopeVerdict.model_validate_json(response.text or "")
        except Exception as e:  # noqa: BLE001
            last = e
    assert last is not None
    raise last
