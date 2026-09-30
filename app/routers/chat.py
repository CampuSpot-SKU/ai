"""backend의 chat.py가 호출하는 의도분류/RAG 엔드포인트."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai.intent import HistoryMessage, IntentClassificationError, IntentResult, classify
from ai.judge import JudgeError, JudgeRequest, JudgeResult, judge
from ai.say import SayError, SayRequest, SayResult, say
from app.deps import verify_internal_secret

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000, examples=["3동 2층 화장실 물이 계속 새요"])
    # 직전 대화 (최근 3개만 사용). 없으면 빈 리스트.
    history: list[HistoryMessage] = Field(default_factory=list)


@router.post("/intent/classify", response_model=IntentResult)
def classify_intent(req: ClassifyRequest) -> IntentResult:
    """발화를 신고(report) / 행정문의(inquiry) / 애매함(unclear)으로 분류.

    unclear면 clarifying_question에 사용자에게 되물을 문장이 들어 있음.
    """
    try:
        return classify(req.text, req.history)
    except IntentClassificationError:
        raise HTTPException(status_code=503, detail="잠시 후 다시 시도해주세요") from None


@router.post("/report/say", response_model=SayResult)
def report_say(req: SayRequest) -> SayResult:
    """신고 접수 대화 문장을 같은 뜻의 자연스러운 말투로 다듬음 (실패하면 503 → backend가 고정 문구 사용)."""
    try:
        return say(req)
    except SayError:
        raise HTTPException(status_code=503, detail="문장 생성 실패") from None


@router.post("/report/judge", response_model=JudgeResult)
def report_judge(req: JudgeRequest) -> JudgeResult:
    """신고 내용의 카테고리·영향도·긴급도·판정 이유를 판정 (실패하면 503 → backend가 규칙 기반 판정 사용)."""
    try:
        return judge(req)
    except JudgeError:
        raise HTTPException(status_code=503, detail="판정 실패") from None


@router.post("/rag/answer")
def rag_answer(payload: dict) -> None:
    # TODO: ai.rag 모듈 호출 (스트리밍 응답은 추후 SSE로 전환) — Phase 1-4
    raise NotImplementedError
