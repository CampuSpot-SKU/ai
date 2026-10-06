"""학교 공지 게시판 수집 (1-14).

학교 홈페이지(WordPress)의 공개 API `/wp-json/wp/v2/notice`에서 공지를 받아
제목·분류·게시일·원문 주소·본문(마크다운)을 정리한다.

- 백필: 정한 날짜 이후 공지를 전부(`collect(since)`) — 결과를 data/notices/notices.jsonl에 저장
- 증분: 마지막 게시일 이후만 받으면 같은 함수로 된다(매일 수집 시 사용)
- 사이트에는 1초에 1번만 요청하고 robots.txt·User-Agent를 지킨다. 읽기 전용.
- 본문은 안내 페이지와 같은 변환기(page_scraper)를 쓴다. 이미지·첨부만 있는 공지는 본문이 거의 비는데,
  그래도 제목으로 검색되도록 `rag.chunk_page`가 따로 다룬다.
"""

from __future__ import annotations

import html as htmllib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ai import page_scraper as ps
from ai import rag

API_PATH = "/wp-json/wp/v2/notice"
KST = timezone(timedelta(hours=9))
PER_PAGE = 50
MAX_API_PAGES = 400  # 안전장치
FIELDS = "id,date,date_gmt,modified,title,link,content,notice-category"
CATEGORIES = {
    41: "일반",
    43: "장학",
    42: "학사",
    45: "대외활동",
    47: "행사안내",
    30: "채용공지",
    48: "FYP",
    46: "등록·납부",
}
NOTICES_PATH = Path(__file__).resolve().parent.parent / "data" / "notices" / "notices.jsonl"

JsonFetcher = Callable[[str], tuple[Any, dict[str, str]]]


class NoticeError(Exception):
    """공지를 못 받았거나 형식이 달라졌을 때."""


def default_since(days: int = 365) -> str:
    """오늘(한국 날짜)에서 days일 전 날짜(YYYY-MM-DD)."""
    return (datetime.now(KST) - timedelta(days=days)).date().isoformat()


def fetch_json(url: str, *, timeout: float = 30.0, retries: int = 2) -> tuple[Any, dict[str, str]]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": ps.USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return json.loads(resp.read().decode("utf-8")), headers
        except urllib.error.HTTPError as e:
            if e.code in (400, 403, 404):  # 400 = 마지막 쪽을 넘어감
                raise NoticeError(f"HTTP {e.code}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise NoticeError(str(last))


def build_url(since: str, page: int, per_page: int = PER_PAGE) -> str:
    query = urllib.parse.urlencode(
        {
            "per_page": per_page,
            "page": page,
            "orderby": "date",
            "order": "asc",
            "after": f"{since}T00:00:00",
            "_fields": FIELDS,
        }
    )
    return f"{ps.BASE_URL}{API_PATH}?{query}"


_JUNK_TAGS = re.compile(r"<!DOCTYPE[^>]*>|<!--.*?-->|</?(?:html|head|body|meta)\b[^>]*>", re.I | re.S)


def clean_html(raw: str) -> str:
    """한글 문서 등을 붙여넣은 공지에 섞인 문서 선언·주석·meta 태그를 지운다(안 지우면 본문에 글자로 나옴)."""
    return _JUNK_TAGS.sub("", raw)


def _title(raw: str) -> str:
    return " ".join(htmllib.unescape(raw).split())


def parse_item(item: dict[str, Any]) -> dict[str, Any]:
    """API 항목 1개 → 저장용 기록."""
    link = str(item.get("link", "")).strip()
    cat_ids = [c for c in item.get("notice-category", []) if isinstance(c, int)]
    category = next((CATEGORIES[c] for c in cat_ids if c in CATEGORIES), "")
    raw_html = clean_html(str((item.get("content") or {}).get("rendered", "")))
    extracted = ps.extract_page(f'<div class="entry-content">{raw_html}</div>', link)
    date_local = str(item.get("date", ""))
    date_gmt = str(item.get("date_gmt", ""))
    return {
        "id": item.get("id"),
        "title": _title(str((item.get("title") or {}).get("rendered", ""))),
        "link": link,
        "category": category,
        "date": date_local,
        "published_at": f"{date_gmt}Z" if date_gmt and not date_gmt.endswith("Z") else date_gmt,
        "modified": str(item.get("modified", "")),
        "body": extracted.body,
        "images": extracted.images,
    }


def collect(
    since: str,
    *,
    fetcher: JsonFetcher | None = None,
    delay: float = 1.0,
) -> list[dict[str, Any]]:
    """since(YYYY-MM-DD) 이후 공지를 오래된 것부터 전부 받는다. 하나라도 실패하면 예외."""
    fetch = fetcher or fetch_json
    if fetcher is None:
        rp = ps.load_robots()
        if not rp.can_fetch(ps.USER_AGENT, f"{ps.BASE_URL}{API_PATH}"):
            raise NoticeError("robots.txt가 공지 수집을 막고 있습니다.")
    records: list[dict[str, Any]] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        if page > MAX_API_PAGES:
            raise NoticeError("쪽 수가 너무 많아 중단했습니다.")
        data, headers = fetch(build_url(since, page))
        if not isinstance(data, list):
            raise NoticeError("응답 형식이 달라졌습니다(목록이 아님).")
        total_pages = int(headers.get("x-wp-totalpages", total_pages) or total_pages)
        for item in data:
            rec = parse_item(item)
            if not rec["link"] or not rec["title"] or not rec["published_at"]:
                raise NoticeError(f"필수 항목이 빈 공지가 있습니다(id {rec['id']}).")
            records.append(rec)
        page += 1
        if fetcher is None and page <= total_pages:
            time.sleep(delay)
    if not records:
        raise NoticeError(f"{since} 이후 공지를 한 건도 못 받았습니다.")
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path = NOTICES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_jsonl(path: Path = NOTICES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def merge(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """주소(link) 기준으로 합친다. 같은 공지는 새 기록으로 덮어쓰고, 게시일 순으로 정렬."""
    by_link = {r["link"]: r for r in old}
    by_link.update({r["link"]: r for r in new})
    return sorted(by_link.values(), key=lambda r: (r["published_at"], r["link"]))


def to_pages(records: list[dict[str, Any]]) -> list[rag.Page]:
    pages = []
    for r in records:
        pages.append(
            rag.Page(
                slug=f"notice-{r['id']}",
                title=r["title"],
                source_url=r["link"],
                body=r["body"],
                doc_type=rag.DOC_TYPE_NOTICE,
                published_at=r["published_at"],
                category=r["category"],
                date_label=str(r["date"])[:10],
            )
        )
    return pages


def summarize(records: list[dict[str, Any]]) -> list[str]:
    """분류별 건수·본문 짧은 공지 수 요약(검수용)."""
    by_cat: dict[str, int] = {}
    short = 0
    for r in records:
        by_cat[r["category"] or "(분류 없음)"] = by_cat.get(r["category"] or "(분류 없음)", 0) + 1
        if rag._readable_chars(r["body"]) < rag.NOTICE_SHORT_CHARS:
            short += 1
    lines = [f"공지 {len(records)}건 ({records[0]['date'][:10]} ~ {records[-1]['date'][:10]})"]
    lines += [f"  {k}: {v}건" for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])]
    lines.append(f"  본문이 거의 없는 공지(제목으로만 검색): {short}건")
    return lines
