"""배치 엔드포인트 유닛테스트 — 인증과 dry_run 전달만 확인 (DB는 가짜로 바꿔 끼움)."""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import cron as cron_mod

client = TestClient(app)
HEADERS = {"X-Internal-Secret": "test-secret"}


@pytest.fixture(autouse=True)
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SERVICE_SECRET", "test-secret")


@contextmanager
def _fake_session() -> Iterator[object]:
    yield object()


def test_batches_require_internal_secret() -> None:
    for path in ("/api/v1/cron/detection-scan", "/api/v1/cron/prediction-update"):
        assert client.post(path).status_code == 401
        assert client.post(path, headers={"X-Internal-Secret": "wrong"}).status_code == 401


def test_detection_scan_passes_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Result:
        def as_dict(self) -> dict[str, Any]:
            return {"clusters_created": 1}

    def fake_run(db: object, dry_run: bool = False) -> _Result:
        seen["dry_run"] = dry_run
        return _Result()

    monkeypatch.setattr(cron_mod, "get_session", _fake_session)
    monkeypatch.setattr(cron_mod, "run_detection", fake_run)
    res = client.post("/api/v1/cron/detection-scan?dry_run=true", headers=HEADERS)
    assert res.status_code == 200
    assert res.json() == {"clusters_created": 1}
    assert seen["dry_run"] is True


def test_prediction_update_passes_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(db: object, dry_run: bool = False) -> dict[str, Any]:
        seen["dry_run"] = dry_run
        return {"predictions": 2}

    monkeypatch.setattr(cron_mod, "get_session", _fake_session)
    monkeypatch.setattr(cron_mod, "run_prediction", fake_run)
    res = client.post("/api/v1/cron/prediction-update", headers=HEADERS)
    assert res.status_code == 200
    assert res.json() == {"predictions": 2}
    assert seen["dry_run"] is False
