"""backend의 chat.py가 호출하는 의도분류/RAG 엔드포인트.

Gemini 실패는 503으로 바꿔 backend가 대체 흐름을 쓰게 하고, 같은 실패를 Sentry에 남김(1-15, app/monitoring.py)."""
import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai.agent import AgentError, TurnRequest, TurnResult, turn
from ai.answer import (
    AnswerError,
    AnswerRequest,
    AnswerResult,
    answer,
    prepare_stream,
    stream_answer,
)
from ai.db import get_session
from ai.intent import HistoryMessage, IntentClassificationError, IntentResult, classify
from ai.judge import JudgeError, JudgeRequest, JudgeResult, judge
from ai.say import SayError, SayRequest, SayResult, say
from app.deps import verify_internal_secret
from app.monitoring import report_failure

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
    except IntentClassificationError as e:
        report_failure("intent", e)
        raise HTTPException(status_code=503, detail="잠시 후 다시 시도해주세요") from None


@router.post("/report/say", response_model=SayResult)
def report_say(req: SayRequest) -> SayResult:
    """신고 접수 대화 문장을 같은 뜻의 자연스러운 말투로 다듬음 (실패하면 503 → backend가 고정 문구 사용)."""
    try:
        return say(req)
    except SayError as e:
        report_failure("say", e)
        raise HTTPException(status_code=503, detail="문장 생성 실패") from None


@router.post("/report/judge", response_model=JudgeResult)
def report_judge(req: JudgeRequest) -> JudgeResult:
    """신고 내용의 카테고리·영향도·긴급도·판정 이유를 판정 (실패하면 503 → backend가 규칙 기반 판정 사용)."""
    try:
        return judge(req)
    except JudgeError as e:
        report_failure("judge", e)
        raise HTTPException(status_code=503, detail="판정 실패") from None


@router.post("/report/turn", response_model=TurnResult)
def report_turn(req: TurnRequest) -> TurnResult:
    """신고 접수 대화 한 턴 — 학생 말을 이해하고 신고 상태·다음 말·다음 행동을 돌려줌 (1-3f).

    실패하면 503 → backend가 기존 규칙 기반 흐름으로 대체.
    """
    try:
        return turn(req)
    except AgentError as e:
        report_failure("agent", e)
        raise HTTPException(status_code=503, detail="대화 처리 실패") from None


@router.post("/rag/answer", response_model=AnswerResult)
def rag_answer(req: AnswerRequest) -> AnswerResult:
    """학칙·안내·공지를 근거로 행정 문의에 답변 + 근거 목록 `sources` (1-4c).

    완성·검증된 답변을 한 번에 돌려준다. 실시간 스트리밍은 `/rag/answer/stream`(1-22)이고, 이쪽은 그 대체 경로로 유지한다.
    실패하면 503 → backend가 "잠시 후 다시 시도해주세요"로 응답.
    """
    try:
        with get_session() as db:
            return answer(req.question, session=db)
    except AnswerError as e:
        report_failure("rag_answer", e)
        raise HTTPException(status_code=503, detail="답변 생성 실패") from None


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _rag_events(question: str) -> Iterator[str]:
    try:
        with get_session() as db:  # 검색·개정 공지 조회만 DB를 쓰고, 스트리밍 중에는 연결을 잡고 있지 않음
            ctx = prepare_stream(question, session=db)
    except Exception as e:  # noqa: BLE001 — 검색(임베딩·DB) 실패 전부
        report_failure("rag_answer_stream", e)
        yield _sse({"error": True})
        return
    for event in stream_answer(ctx, on_error=lambda e: report_failure("rag_answer_stream", e)):
        yield _sse(event)


@router.post("/rag/answer/stream")
def rag_answer_stream(req: AnswerRequest) -> StreamingResponse:
    """`/rag/answer`와 같은 요청으로 답변을 **실시간 스트리밍**(SSE, 1-22, 명세 5-1).

    `data: {"delta": "글자"}` 여러 번 → `data: {"done": true, "sources": [...]}`.
    첫 글자 전에 실패하면 `data: {"error": true}` 한 줄 — backend가 `/rag/answer`로 대체 호출한다.
    """
    return StreamingResponse(
        _rag_events(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
