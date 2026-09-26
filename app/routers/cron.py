"""Cloud Scheduler가 이 AI 서비스에 직접 호출하는 배치 엔드포인트
(반복패턴 탐지·재발예측·공지크롤링은 AI 서비스 도메인이라 여기로 옮김 —
SLA 체크는 reports 도메인이라 backend/app/routers/cron.py에 그대로 남음)"""
from fastapi import APIRouter, Depends

from app.deps import verify_internal_secret

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


@router.post("/cron/detection-scan")
def detection_scan():
    # TODO: ai.detection 모듈 호출
    raise NotImplementedError


@router.post("/cron/prediction-update")
def prediction_update():
    # TODO: ai.prediction 모듈 호출
    raise NotImplementedError


@router.post("/cron/crawl-notices")
def crawl_notices():
    # TODO: ai.notice_crawler 모듈 호출
    raise NotImplementedError
