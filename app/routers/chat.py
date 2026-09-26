"""backend의 chat.py가 호출하는 의도분류/RAG 엔드포인트 — TODO: ai.intent/ai.rag 로직 연결"""
from fastapi import APIRouter, Depends

from app.deps import verify_internal_secret

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


@router.post("/intent/classify")
def classify_intent(payload: dict):
    # TODO: ai.intent 모듈 호출
    raise NotImplementedError


@router.post("/rag/answer")
def rag_answer(payload: dict):
    # TODO: ai.rag 모듈 호출 (스트리밍 응답은 추후 SSE로 전환)
    raise NotImplementedError
