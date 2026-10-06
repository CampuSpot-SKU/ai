from __future__ import annotations

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
