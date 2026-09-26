"""캠퍼스팟 AI 서비스 — 별도 Cloud Run 서비스로 배포됨.
backend는 이 서비스를 HTTP로 호출 (내부 인증은 X-Internal-Secret 헤더, AI_SERVICE_SECRET과 대조).
"""
from fastapi import FastAPI

from app.routers import chat, cron

app = FastAPI(title="campuspot-ai")

app.include_router(chat.router, prefix="/api/v1")
app.include_router(cron.router, prefix="/api/v1")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
