"""캠퍼스팟 AI 서비스 — 별도 Cloud Run 서비스로 배포됨.
backend는 이 서비스를 HTTP로 호출 (내부 인증은 X-Internal-Secret 헤더, AI_SERVICE_SECRET과 대조).
"""
from fastapi import FastAPI

from app.routers import chat, cron

app = FastAPI(title="campuspot-ai")

app.include_router(chat.router, prefix="/api/v1")
app.include_router(cron.router, prefix="/api/v1")


# 주의: Cloud Run은 z로 끝나는 경로(/healthz 등)를 예약해서 외부에서 404가 남 → /health 사용
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
