"""Cloud Scheduler가 이 AI 서비스에 직접 호출하는 배치 엔드포인트
(반복패턴 탐지·재발예측·공지크롤링은 AI 서비스 도메인이라 여기로 옮김 —
SLA 체크는 reports 도메인이라 backend/app/routers/cron.py에 그대로 남음)

- 모든 호출에 `X-Internal-Secret` 헤더가 필요하다(Cloud Scheduler의 HTTP 헤더에 넣는다).
- `dry_run=true`를 붙이면 계산만 하고 DB에는 쓰지 않는다(결과 숫자만 돌려줌) — 배포 후 확인용.
"""
from typing import Any

from fastapi import APIRouter, Depends

from ai.db import get_session
from ai.detection import run_detection
from ai.prediction import run_prediction
from app.deps import verify_internal_secret

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


@router.post("/cron/detection-scan")
def detection_scan(dry_run: bool = False) -> dict[str, Any]:
    """반복 신고를 문제 후보로 묶는다(1-8). 매일 1회."""
    with get_session() as db:
        return run_detection(db, dry_run=dry_run).as_dict()


@router.post("/cron/prediction-update")
def prediction_update(dry_run: bool = False) -> dict[str, Any]:
    """재발 주기 통계와 다음 예상 시점을 다시 계산한다(1-11). 매일 1회."""
    with get_session() as db:
        return run_prediction(db, dry_run=dry_run)


@router.post("/cron/crawl-notices")
def crawl_notices():
    # TODO: ai.notice_crawler 모듈 호출
    raise NotImplementedError
