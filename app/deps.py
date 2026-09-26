"""공통 의존성 — 내부 호출 인증(X-Internal-Secret) 등."""
import os

from fastapi import Header, HTTPException


def verify_internal_secret(x_internal_secret: str = Header(default="")):
    expected = os.environ.get("AI_SERVICE_SECRET", "")
    if not expected or x_internal_secret != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-Internal-Secret")
