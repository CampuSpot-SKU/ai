"""실시간 스트리밍 답변(1-22) 유닛테스트 — Gemini·DB는 가짜로 바꿔 끼움."""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai import answer as ans
from ai import rag
from app.main import app
from app.routers import chat as chat_router

ANSWER = "학칙 제29조에 따르면 휴학은 학기 시작 전에 신청할 수 있어요."


def _hit(**kw: Any) -> rag.Hit:
    base: dict[str, Any] = {
        "text": "휴학 안내 " * 20, "title": "학칙 제29조(휴학)", "doc_type": "학칙", "article_no": "제29조",
        "url": "https://x.kr/r#29", "dist": 0.2, "published_at": None,
    }
    base.update(kw)
    return rag.Hit(base["text"], base["title"], base["doc_type"], base["article_no"], base["url"], base["dist"],
                   base["published_at"])


def _json(answer: str = ANSWER, used: list[int] | None = None, answerable: bool = True) -> str:
    return json.dumps({"answerable": answerable, "answer": answer, "used": [1] if used is None else used},
                      ensure_ascii=False)


def _split(raw: str, size: int) -> list[str]:
    return [raw[i : i + size] for i in range(0, len(raw), size)]


def _fake_stream(monkeypatch: pytest.MonkeyPatch, *attempts: list[str] | Exception) -> list[int]:
    """시도마다 조각 목록(또는 시작 전 예외)을 돌려주는 가짜 Gemini — 호출 횟수를 담은 리스트를 돌려줌."""
    calls: list[int] = []
    queue = list(attempts)

    def fake(question: str, hits: list[rag.Hit]) -> Iterator[str]:
        calls.append(1)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        yield from item

    monkeypatch.setattr(ans, "_stream_gemini", fake)
    return calls


def _ctx(**kw: Any) -> ans.StreamContext:
    return ans.StreamContext("휴학", kw.get("hits", [_hit()]), kw.get("notices"))


def _run(ctx: ans.StreamContext, **kw: Any) -> list[dict[str, Any]]:
    return list(ans.stream_answer(ctx, **kw))


def _text(events: list[dict[str, Any]]) -> str:
    return "".join(e.get("delta", "") for e in events)


def test_answer_so_far_waits_for_split_escapes() -> None:
    raw = '{"answerable": true, "answer": "줄\\n바꿈 \\"인용\\" \\u0041\\ud83d\\ude00끝", "used": [1]}'
    assert ans._answer_so_far(raw) == (True, '줄\n바꿈 "인용" A😀끝', True)
    # 이스케이프 한가운데서 잘린 조각은 그 앞까지만 해독하고, 문자열이 닫히지 않은 것으로 본다
    for cut in range(len(raw)):
        ok, text, closed = ans._answer_so_far(raw[:cut])
        assert ok in (None, True)
        assert '줄\n바꿈 "인용" A😀끝'.startswith(text)
        assert closed is False or cut >= raw.index('끝"') + 2
    assert ans._answer_so_far('{"answerable": false, "answer": "", "used": []}')[:2] == (False, "")
    assert ans._answer_so_far('{"answer": "먼저 온 답", "answerable": tr')[0] is None


def test_streams_text_in_pieces_then_done_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_stream(monkeypatch, _split(_json(), 7))
    events = _run(_ctx())
    deltas = [e for e in events if "delta" in e]
    assert len(deltas) > 3 and _text(events) == ANSWER
    assert events[-1] == {
        "done": True,
        "sources": [{"title": "학칙 제29조(휴학)", "article_no": "제29조", "url": "https://x.kr/r#29", "as_of": None}],
    }
    assert all("error" not in e for e in events)


def test_no_text_is_sent_before_answerable_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = '{"answer": "' + ANSWER + '", "used": [1], "answerable": true}'  # 순서가 바뀌어도 answerable 전엔 안 보냄
    _fake_stream(monkeypatch, [raw[:30], raw[30:-12], raw[-12:]])
    events = _run(_ctx())
    assert _text(events) == ANSWER and events[-1]["done"] is True


def test_markdown_characters_are_removed_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_stream(monkeypatch, _split(_json("**휴학**은 `학기 시작 전`에 ## 신청해요."), 3))
    events = _run(_ctx())
    assert _text(events) == "휴학은 학기 시작 전에  신청해요."
    assert events[-1]["sources"]  # 서식을 지웠어도 정상 완료


def test_unanswerable_sends_notice_once_without_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_stream(monkeypatch, _split(_json("", [], answerable=False), 5))
    events = _run(_ctx())
    assert events == [{"delta": ans.NO_EVIDENCE_ANSWER}, {"done": True, "sources": []}]


def test_no_hits_skips_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_stream(monkeypatch)
    events = _run(_ctx(hits=[]))
    assert events == [{"delta": ans.NO_EVIDENCE_ANSWER}, {"done": True, "sources": []}] and calls == []


@pytest.mark.parametrize("used", [[], [9], [0]])
def test_wrong_citation_after_streaming_appends_notice_and_drops_sources(
    monkeypatch: pytest.MonkeyPatch, used: list[int]
) -> None:
    _fake_stream(monkeypatch, _split(_json(used=used), 9))
    events = _run(_ctx())
    assert _text(events) == ANSWER + ans.NO_CITATION_NOTE
    assert events[-1] == {"done": True, "sources": []}


def test_failure_before_first_text_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_stream(monkeypatch, RuntimeError("quota"), _split(_json(), 11))
    events = _run(_ctx())
    assert len(calls) == 2 and _text(events) == ANSWER and events[-1]["done"] is True


def test_two_failures_end_with_error_event_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_stream(monkeypatch, RuntimeError("a"), ["not json at all"])
    seen: list[Exception] = []
    events = _run(_ctx(), on_error=seen.append)
    assert events == [{"error": True}] and len(calls) == 2 and len(seen) == 1


def test_failure_after_first_text_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(question: str, hits: list[rag.Hit]) -> Iterator[str]:
        yield '{"answerable": true, "answer": "학칙 제29조에 따르면 '
        raise ConnectionError("cut")

    calls: list[int] = []
    monkeypatch.setattr(ans, "_stream_gemini", lambda q, h: (calls.append(1), broken(q, h))[1])
    events = _run(_ctx())
    assert len(calls) == 1
    assert _text(events) == "학칙 제29조에 따르면" + ans.CUT_OFF_NOTE
    assert events[-1] == {"done": True, "sources": []}


def test_text_is_capped_at_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_stream(monkeypatch, _split(_json("가" * (ans.MAX_ANSWER_LEN + 200)), 100))
    events = _run(_ctx())
    assert _text(events) == "가" * ans.MAX_ANSWER_LEN + ans.CUT_OFF_NOTE
    assert events[-1] == {"done": True, "sources": []}


def test_revision_notice_is_sent_as_extra_text_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_stream(monkeypatch, _split(_json(), 13))
    hit = _hit(published_at="2025-10-01T00:00:00+00:00")
    notice = ans.RevisionNotice("학칙 일부개정안 사전 공고", "https://x.kr/n", ans.datetime(2026, 1, 19, tzinfo=ans.UTC))
    events = _run(_ctx(hits=[hit], notices=[notice]))
    text = _text(events)
    assert text.startswith(ANSWER) and "사전 공고가 있었어요" in text
    assert [s["title"] for s in events[-1]["sources"]] == ["학칙 제29조(휴학)", "학칙 일부개정안 사전 공고"]


def test_prepare_stream_skips_notice_lookup_without_session_and_survives_failure() -> None:
    hits = [_hit()]
    def searcher(q: str, **_k: Any) -> list[rag.Hit]:
        return hits

    assert ans.prepare_stream("휴학", session=None, searcher=searcher).notices is None

    def boom(_s: Any) -> list[ans.RevisionNotice]:
        raise RuntimeError("db down")

    ctx = ans.prepare_stream("휴학", session=object(), searcher=searcher, revisions=boom)
    assert ctx.hits and ctx.notices is None
    assert ans.prepare_stream("휴학", session=object(), searcher=searcher, revisions=lambda s: []).notices == []


# ---------------------------------------------------------------- 엔드포인트

client = TestClient(app)
HEADERS = {"X-Internal-Secret": "test-secret"}


@pytest.fixture(autouse=True)
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_SERVICE_SECRET", "test-secret")


@contextmanager
def _fake_session() -> Iterator[object]:
    yield object()


def _events(res: Any) -> list[dict[str, Any]]:
    return [json.loads(b[5:].strip()) for b in res.text.split("\n\n") if b.startswith("data:")]


def test_stream_endpoint_requires_secret_and_streams_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    assert client.post("/api/v1/rag/answer/stream", json={"question": "휴학"}).status_code == 401
    monkeypatch.setattr(chat_router, "get_session", _fake_session)
    monkeypatch.setattr(chat_router, "prepare_stream", lambda q, session: _ctx())
    _fake_stream(monkeypatch, _split(_json(), 10))
    res = client.post("/api/v1/rag/answer/stream", json={"question": "휴학"}, headers=HEADERS)
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    events = _events(res)
    assert "".join(e.get("delta", "") for e in events) == ANSWER and events[-1]["done"] is True


def test_stream_endpoint_reports_search_failure_as_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(q: str, session: Any) -> ans.StreamContext:
        raise RuntimeError("embedding down")

    reported: list[str] = []
    monkeypatch.setattr(chat_router, "get_session", _fake_session)
    monkeypatch.setattr(chat_router, "prepare_stream", boom)
    monkeypatch.setattr(chat_router, "report_failure", lambda where, e: reported.append(where))
    res = client.post("/api/v1/rag/answer/stream", json={"question": "휴학"}, headers=HEADERS)
    assert _events(res) == [{"error": True}] and reported == ["rag_answer_stream"]
