from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ai import notice_crawler as nc
from ai import rag


def _item(i: int, title: str, html: str, cat: int = 42, day: str = "2026-10-01") -> dict[str, Any]:
    return {
        "id": i,
        "date": f"{day}T10:00:00",
        "date_gmt": f"{day}T01:00:00",
        "modified": f"{day}T10:00:00",
        "title": {"rendered": title},
        "link": f"https://www.skuniv.ac.kr/notice/{i}",
        "content": {"rendered": html},
        "notice-category": [cat],
    }


def test_parse_item_unescapes_title_and_converts_body() -> None:
    rec = nc.parse_item(
        _item(7, "취업 &amp; 현장실습 설명회", '<p>일시: <strong>10월 7일</strong></p><img src="a.png">')
    )
    assert rec["title"] == "취업 & 현장실습 설명회"
    assert rec["category"] == "학사"
    assert rec["published_at"] == "2026-10-01T01:00:00Z"
    assert "10월 7일" in rec["body"] and rec["images"] == 1


def test_collect_pages_and_rejects_empty() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> tuple[Any, dict[str, str]]:
        calls.append(url)
        if "page=1" in url:
            return [_item(1, "가", "<p>본문</p>")], {"x-wp-totalpages": "2"}
        return [_item(2, "나", "<p>본문</p>", cat=43, day="2026-10-02")], {"x-wp-totalpages": "2"}

    recs = nc.collect("2025-10-06", fetcher=fetcher)
    assert [r["id"] for r in recs] == [1, 2] and len(calls) == 2
    assert recs[1]["category"] == "장학"

    with pytest.raises(nc.NoticeError):
        nc.collect("2025-10-06", fetcher=lambda u: ([], {"x-wp-totalpages": "1"}))
    with pytest.raises(nc.NoticeError):
        nc.collect("2025-10-06", fetcher=lambda u: ({"code": "x"}, {}))


def test_merge_overwrites_by_link_and_sorts(tmp_path: Any) -> None:
    a = nc.parse_item(_item(1, "가", "<p>옛</p>", day="2026-09-01"))
    b = nc.parse_item(_item(2, "나", "<p>본문</p>", day="2026-10-01"))
    a2 = dict(a, body="새 본문")
    merged = nc.merge([b, a], [a2])
    assert [r["id"] for r in merged] == [1, 2] and merged[0]["body"] == "새 본문"
    path = tmp_path / "n.jsonl"
    nc.save_jsonl(merged, path)
    assert nc.load_jsonl(path) == merged


def test_to_pages_and_summary() -> None:
    long_html = "<p>" + "수강신청 안내 내용입니다. " * 20 + "</p>"
    recs = [
        nc.parse_item(_item(1, "수강신청", long_html)),
        nc.parse_item(_item(2, "포스터만", '<img src="p.png">', cat=41)),
    ]
    pages = nc.to_pages(recs)
    assert pages[0].doc_type == rag.DOC_TYPE_NOTICE and pages[0].date_label == "2026-10-01"
    lines = nc.summarize(recs)
    assert "공지 2건" in lines[0] and "1건" in lines[-1]


def test_doctype_and_meta_junk_is_removed() -> None:
    html = (
        '<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN" '
        '"http://www.w3.org/TR/REC-html40/loose.dtd"><html><head><meta charset="utf-8"></head>'
        "<body><!-- x --><p>독감 예방접종 실시 안내</p></body></html>"
    )
    rec = nc.parse_item(_item(9, "독감", html))
    assert "DTD" not in rec["body"] and "PUBLIC" not in rec["body"]
    assert "독감 예방접종 실시 안내" in rec["body"]


# ---------------------------------------------------------------- 매일 증분 수집 (1-14)


def test_since_from_latest_uses_korean_date_of_latest_notice() -> None:
    # UTC 10/8 16:00 = 한국 10/9 01:00 → 한국 날짜로 10/9부터 다시 받는다
    assert nc.since_from_latest(datetime(2026, 10, 8, 16, 0, tzinfo=UTC)) == "2026-10-09"
    naive = datetime(2026, 10, 8, 10, 0)  # noqa: DTZ001 — 시간대 없는 값도 UTC로 본다
    assert nc.since_from_latest(naive) == "2026-10-08"


def test_since_from_latest_without_notices_falls_back() -> None:
    assert nc.since_from_latest(None, backfill_days=30) == nc.default_since(30)


def test_collect_allow_empty() -> None:
    def fetcher(url: str) -> tuple[Any, dict[str, str]]:
        return [], {"x-wp-totalpages": "1"}

    assert nc.collect("2026-10-09", fetcher=fetcher, allow_empty=True) == []
    with pytest.raises(nc.NoticeError):
        nc.collect("2026-10-09", fetcher=fetcher)


class _Scalar:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar(self) -> Any:
        return self.value


class _FakeDb:
    def execute(self, *_a: Any, **_k: Any) -> _Scalar:
        return _Scalar(datetime(2026, 10, 1, 1, 0, tzinfo=UTC))


def _one_notice_fetcher(url: str) -> tuple[Any, dict[str, str]]:
    return [_item(9, "새 공지", "<p>본문입니다 본문입니다 본문입니다</p>", day="2026-10-02")], {
        "x-wp-totalpages": "1"
    }


def test_run_notice_crawl_dry_run_only_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("dry_run에서는 DB에 쓰면 안 됨")

    monkeypatch.setattr(rag, "sync_documents", boom)
    res = nc.run_notice_crawl(_FakeDb(), dry_run=True, fetcher=_one_notice_fetcher)
    assert res == {"since": "2026-10-01", "fetched": 1, "dry_run": True}


def test_run_notice_crawl_syncs_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_sync(pages: Any, **kw: Any) -> rag.SyncResult:
        seen["titles"] = [p.title for p in pages]
        seen["doc_type"] = kw["doc_type"]
        return rag.SyncResult(added=1, chunks=2)

    monkeypatch.setattr(rag, "sync_documents", fake_sync)
    res = nc.run_notice_crawl(_FakeDb(), fetcher=_one_notice_fetcher)
    assert seen == {"titles": ["새 공지"], "doc_type": rag.DOC_TYPE_NOTICE}
    assert res["added"] == 1 and res["chunks"] == 2 and res["unchanged"] == 0


def test_run_notice_crawl_no_new_notices_skips_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "sync_documents", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    res = nc.run_notice_crawl(_FakeDb(), fetcher=lambda u: ([], {"x-wp-totalpages": "1"}))
    assert res["fetched"] == 0 and "added" not in res
