"""Sentry 연결 테스트 (S-3·1-15) — 실제 Sentry로 보내지 않고 가짜 전송으로 이벤트 모양만 확인."""
from typing import Any

import sentry_sdk
from fastapi.testclient import TestClient

from app.monitoring import report_failure


class _Box(sentry_sdk.transport.Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def capture_envelope(self, envelope: Any) -> None:
        for item in envelope.items:
            if item.type == "event":
                self.events.append(item.payload.json)


def test_report_failure_tags_and_hides_locals() -> None:
    box = _Box()
    sentry_sdk.init(dsn="https://k@o0.ingest.sentry.io/0", transport=box, send_default_pii=False,
                    include_local_variables=False)
    try:
        student_text = "학생 문장"  # noqa: F841 — 지역 변수로만 있음 → 보내지 않아야 함
        try:
            raise RuntimeError("Gemini 429")
        except RuntimeError as e:
            report_failure("intent", e)
        sentry_sdk.flush()
    finally:
        sentry_sdk.init(dsn=None)
    assert len(box.events) == 1
    ev = box.events[0]
    assert ev["tags"]["ai_failure"] == "intent"
    frames = [f for v in ev["exception"]["values"] for f in v["stacktrace"]["frames"]]
    assert frames and all("vars" not in f for f in frames)  # 지역 변수(학생 문장 등)를 안 보냄


def test_init_without_dsn_is_noop(monkeypatch: Any) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    from app.main import app  # init_sentry가 DSN 없이도 앱을 띄움

    assert TestClient(app).get("/health").json() == {"status": "ok"}
