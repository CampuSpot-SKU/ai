"""학교 홈페이지 '안내 페이지' 수집·추출 도구 (작업 1-4a).

공지가 아니라 홈페이지 메뉴에 있는 안내(수강신청·휴학·장학·학생증 등)를 RAG 원본으로 쓰기 위해
페이지마다 본문만 뽑아 마크다운(`data/pages/<slug>.md`)으로 저장한다.

- 본문 영역: 워드프레스 `article .entry-content` (메뉴·푸터·검색창 같은 반복 요소는 제외)
- 표: 마크다운 표로 바꾸되 rowspan·colspan을 풀어서 행마다 뜻이 독립적으로 읽히게 한다
- 본문이 비었거나(동적 위젯·이미지뿐) 짧으면 상태(`empty`·`short`)로 표시 → 사람이
  `data/pages/manual/<slug>.md`에 복붙하면 그 파일이 우선한다 (스크린샷이면 Claude가 옮겨 적음)
- 같은 내용이 다른 주소로 열리면 `duplicate`로 표시
- 내용이 안 바뀌면(content_hash 동일) 파일을 다시 쓰지 않아 git diff가 조용하다
  → 나중에 주기적으로 다시 돌려도(1-14와 같은 방식) 바뀐 페이지만 diff로 보인다

HTTP는 표준 라이브러리(urllib)만 쓴다. 호출 간격·robots.txt·User-Agent를 지킨다.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

BASE_URL = "https://www.skuniv.ac.kr"
SITE_HOST = "www.skuniv.ac.kr"
USER_AGENT = "CampuSpotBot/1.0 (+university student contest project; 1 request/sec)"

MAX_PAGES = 30  # 게시판형 페이지를 따라갈 최대 쪽 수 (안전장치)
OK_MIN_CHARS = 300  # 이 이상이면 정상
EMPTY_MAX_CHARS = 50  # 이 미만이면 본문 없음

_SITE_TITLE_SUFFIX = re.compile(r"\s*[-|–]\s*서경대학교\s*$")
_DROP_SELECTORS = (
    ".sub-menu-wrap", ".sub-page-menu-wrap", ".post-edit-link-wrap", ".screen-reader-text", ".sr-only",
    ".board-list-header", ".pagination-wrap",
)
_NEW_WINDOW = re.compile(r"\s*\(새 창 열림\)")
_DROP = {"script", "style", "noscript", "svg", "form", "button", "nav", "aside", "template", "head"}
_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_INLINE = {
    "a", "span", "strong", "b", "em", "i", "u", "small", "sup", "sub", "code", "label", "font",
    "mark", "abbr", "time", "s", "del", "ins", "cite", "q",
}
_BLOCK = {
    "p", "div", "ul", "ol", "li", "table", "section", "article", "blockquote", "figure", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "dl", "dt", "dd", "hr", "iframe",
}

Fetcher = Callable[[str], "tuple[str, str]"]


class FetchError(Exception):
    """페이지를 못 가져옴 (404·접속 실패 등)."""


# ---------------------------------------------------------------- HTTP

def fetch_html(url: str, *, timeout: float = 20.0, retries: int = 2) -> tuple[str, str]:
    """(html, 최종 주소). 리다이렉트는 따라간다. 404·403은 재시도 없이 바로 실패."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ko"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="replace"), str(resp.geturl())
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                raise FetchError(f"HTTP {e.code}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise FetchError(str(last))


def load_robots(
    base_url: str = BASE_URL, fetcher: Fetcher | None = None
) -> robotparser.RobotFileParser:
    """robots.txt를 읽는다. 못 읽으면(없음 포함) 전체 허용으로 본다."""
    rp = robotparser.RobotFileParser()
    try:
        text, _ = (fetcher or fetch_html)(urljoin(base_url, "/robots.txt"))
        rp.parse(text.splitlines())
    except FetchError:
        rp.parse([])
    return rp


def discover_sitemap(fetcher: Fetcher | None = None) -> list[str]:
    """사이트맵의 모든 페이지 주소 (퍼센트 인코딩은 그대로)."""
    xml, _ = (fetcher or fetch_html)(urljoin(BASE_URL, "/page-sitemap.xml"))
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


# ---------------------------------------------------------------- HTML → 마크다운

def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = _NEW_WINDOW.sub("", text)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def _span(cell: Tag, attr: str, limit: int) -> int:
    raw = str(cell.get(attr, "1")).strip()
    try:
        value = int(raw)
    except ValueError:
        return 1
    return min(max(value, 1), limit)


class _Converter:
    def __init__(self, base_url: str, heading_offset: int) -> None:
        self.base_url = base_url
        self.offset = heading_offset
        self.headings: list[str] = []
        self.tables = 0
        self.images = 0
        self.embeds: list[str] = []
        self.faq_items = 0
        self._in_faq = False

    # --- 인라인
    def inline(self, el: Tag, skip: tuple[str, ...] = ()) -> str:
        return _clean("".join(self._node(c, skip) for c in el.children))

    def _node(self, c: object, skip: tuple[str, ...] = ()) -> str:
        if isinstance(c, Comment):
            return ""
        if isinstance(c, NavigableString):
            return str(c)
        if not isinstance(c, Tag) or c.name in _DROP or c.name in skip:
            return ""
        if c.name == "br":
            return "\n"
        if c.name == "img":
            return self._img(c)
        if c.name == "iframe":
            return self._embed(c)
        if c.name == "a":
            return self._link(c)
        if c.name in _BLOCK:
            return " " + self.inline(c) + " "
        return self.inline(c)

    def _link(self, a: Tag) -> str:
        text = self.inline(a)
        href = str(a.get("href", "")).strip()
        if not text or not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return text
        url = urljoin(self.base_url, href)
        if not url.startswith("http") or url.rstrip("/") == text.rstrip("/"):
            return text
        return f"[{text}]({url})"

    def _img(self, img: Tag) -> str:
        self.images += 1
        alt = _clean(str(img.get("alt", "")))
        return f"[이미지: {alt}]" if alt else "[이미지]"

    def _embed(self, frame: Tag) -> str:
        src = urljoin(self.base_url, str(frame.get("src", "")).strip())
        self.embeds.append(src)
        return f"[임베드: {src}]"

    # --- 블록
    def blocks(self, el: Tag) -> list[str]:
        out: list[str] = []
        buf: list[str] = []

        def flush() -> None:
            text = _clean("".join(buf))
            buf.clear()
            if text:
                out.append(text)

        for child in el.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                buf.append(str(child))
                continue
            if not isinstance(child, Tag) or child.name in _DROP:
                continue
            name = child.name
            if name in _INLINE and not child.find(list(_BLOCK)):
                buf.append(self._node(child))
                continue
            if name == "img":
                buf.append(self._img(child))
                continue
            flush()
            if name == "br" or name == "hr":
                continue
            if name == "div" and "accordion" in (child.get("class") or []) \
                    and child.select_one(".accordion-content"):
                out.append(self._faq(child))
                continue
            if name == "a":  # 카드처럼 블록을 감싼 링크: 안쪽 내용 뒤에 주소를 붙인다
                out.extend(self.blocks(child))
                href = str(child.get("href", "")).strip()
                url = urljoin(self.base_url, href) if href else ""
                if url.startswith("http") and not href.startswith("#"):
                    out.append(f"링크: {url}")
                continue
            if name in _HEADINGS and self._in_faq:
                text = self.inline(child).replace("\n", " ")
                if text:
                    out.append(f"**{text}**")
            elif name in _HEADINGS:
                text = self.inline(child).replace("\n", " ")
                if text:
                    level = min(max(_HEADINGS[name] + self.offset, 2), 6)
                    out.append("#" * level + " " + text)
                    self.headings.append(text)
            elif name == "p":
                text = self.inline(child)
                strong = _clean(" ".join(x.get_text(" ") for x in child.find_all(["strong", "b"])))
                if text and strong and strong == text.replace("\n", " ") and len(text) <= 100:
                    text = f"**{text}**"  # 문단 전체가 굵은 글씨면 소제목 역할이라 표시를 남긴다
                if text:
                    out.append(text)
            elif name in ("ul", "ol"):
                block = self._list(child, 0)
                if block:
                    out.append(block)
            elif name == "table":
                block = self._table(child)
                if block:
                    out.append(block)
            elif name == "blockquote":
                inner = "\n\n".join(self.blocks(child))
                if inner:
                    out.append("\n".join("> " + ln for ln in inner.split("\n")))
            elif name == "iframe":
                out.append(self._embed(child))
            elif name == "pre":
                text = child.get_text("\n").strip("\n")
                if text.strip():
                    out.append(text)
            else:  # div·section·article·figure·dl·details ... 안으로 들어감
                out.extend(self.blocks(child))
        flush()
        return out

    def _faq(self, item: Tag) -> str:
        """FAQ 아코디언 한 항목 → `### Q. 제목` + 분류·부서·날짜 + `A.` 답변 (질문 단위로 청킹되게)."""
        self.faq_items += 1
        title_el = item.select_one(".post-title")
        title = self.inline(title_el).replace("\n", " ") if title_el else ""
        category_el = item.select_one(".category")
        category = self.inline(category_el).replace("\n", " ") if category_el else ""
        info_el = item.select_one(".post-info-wrap")
        info = [self.inline(d).replace("\n", " ") for d in info_el.find_all("div", recursive=False)
                if "divider" not in (d.get("class") or [])] if info_el else []
        meta = " · ".join(x for x in [category, *info] if x)
        content = item.select_one(".post-content")
        if content is not None:
            for label in content.select(".answer, .question"):
                label.decompose()
        self._in_faq = True
        try:
            answer = "\n\n".join(self.blocks(content)) if content is not None else ""
        finally:
            self._in_faq = False
        self.headings.append(f"Q. {title}")
        parts = [f"### Q. {title}"]
        if meta:
            parts.append(f"({meta})")
        if answer:
            parts.append(f"A. {answer}")
        return "\n\n".join(parts)

    def _list(self, lst: Tag, depth: int) -> str:
        lines: list[str] = []
        ordered = lst.name == "ol"
        for number, li in enumerate(lst.find_all("li", recursive=False), start=1):
            text = self.inline(li, skip=("ul", "ol")).replace("\n", " ")
            marker = f"{number}. " if ordered else "- "
            if text:
                lines.append("  " * depth + marker + text)
            for sub in li.find_all(["ul", "ol"], recursive=False):
                nested = self._list(sub, depth + 1)
                if nested:
                    lines.append(nested)
        return "\n".join(lines)

    def _cell_text(self, cell: Tag) -> str:
        parts = [b.replace("\n", " ") for b in self.blocks(cell)]
        return " / ".join(p for p in parts if p).replace("|", "\\|")

    def _table(self, tbl: Tag) -> str:
        self.tables += 1
        rows = [tr for tr in tbl.find_all("tr") if tr.find_parent("table") is tbl]
        grid: dict[tuple[int, int], str] = {}
        for r, tr in enumerate(rows):
            c = 0
            for cell in tr.find_all(["th", "td"], recursive=False):
                while (r, c) in grid:
                    c += 1
                text = self._cell_text(cell)
                rs = _span(cell, "rowspan", len(rows) - r)
                cs = _span(cell, "colspan", 30)
                for dr in range(rs):
                    for dc in range(cs):
                        grid[(r + dr, c + dc)] = text
                c += cs
        if not grid:
            return ""
        height = max(r for r, _ in grid) + 1
        width = max(c for _, c in grid) + 1
        matrix = [[grid.get((r, c), "") for c in range(width)] for r in range(height)]
        matrix = [row for row in matrix if any(row)]
        if not matrix:
            return ""
        lines = ["| " + " | ".join(matrix[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in matrix[1:]]
        return "\n".join(lines)


@dataclass
class Extracted:
    title: str
    body: str
    headings: list[str] = field(default_factory=list)
    tables: int = 0
    images: int = 0
    embeds: list[str] = field(default_factory=list)
    repeated_removed: int = 0
    faq_items: int = 0


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.get_text(strip=True):
        title = _SITE_TITLE_SUFFIX.sub("", soup.title.get_text(strip=True)).strip()
        if title:
            return title
    h2 = soup.select_one("header#masthead h2.title")
    return h2.get_text(strip=True) if h2 else ""


def _norm_title(soup: BeautifulSoup) -> str:
    h2 = soup.select_one("header#masthead h2.title")
    return h2.get_text(strip=True) if h2 else ""


def _find_root(soup: BeautifulSoup) -> Tag | None:
    for selector in ("article .entry-content", ".entry-content", "main article", "main"):
        el = soup.select_one(selector)
        if el is not None:
            return el
    return None


def _norm_line(line: str) -> str:
    """중복 비교용: 글머리·번호·공백·기호 차이를 무시한다."""
    return re.sub(r"[\s•·\-*※\d.)(]+", "", line)


def drop_repeated_lines(body: str, min_len: int = 40) -> tuple[str, int]:
    """사이트가 PC·모바일용으로 같은 문장을 두 번 넣은 경우가 있어, 긴 줄의 두 번째부터 지운다.
    표(`|`로 시작)는 건드리지 않는다. (지운 줄 수)도 돌려준다."""
    seen: set[str] = set()
    removed = 0
    out: list[str] = []
    for line in body.split("\n"):
        key = _norm_line(line)
        if not line.startswith("|") and not line.startswith("#") and len(key) >= min_len:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        out.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return text, removed


def extract_page(html: str, url: str) -> Extracted:
    """HTML에서 본문을 뽑아 마크다운으로 바꾼다 (제목 H1은 저장할 때 붙임)."""
    soup = BeautifulSoup(html, "html.parser")
    title = _page_title(soup)
    root = _find_root(soup)
    if root is None:
        return Extracted(title=title, body="")
    for selector in _DROP_SELECTORS:
        for el in root.select(selector):
            el.decompose()
    levels = [
        _HEADINGS[h.name] for h in root.find_all(list(_HEADINGS))
        if h.get_text(strip=True) and h.find_parent(class_="accordion-content") is None
    ]
    offset = 2 - min(levels) if levels else 0
    conv = _Converter(url, offset)
    blocks = conv.blocks(root)
    deduped = [b for i, b in enumerate(blocks) if i == 0 or b != blocks[i - 1]]
    body = re.sub(r"\n{3,}", "\n\n", "\n\n".join(deduped)).strip()
    removed = 0
    if not conv.faq_items:  # FAQ는 항목마다 같은 안내 문구가 있을 수 있어 중복 제거를 하지 않는다
        body, removed = drop_repeated_lines(body)
    first, _, rest = body.partition("\n")
    if first.strip() and first.strip() in (title, _norm_title(soup)):
        body = rest.strip()
    return Extracted(
        title, body, conv.headings, conv.tables, conv.images, conv.embeds, removed, conv.faq_items
    )


def find_last_page(html: str) -> int:
    """게시판형 페이지(FAQ 등)의 마지막 쪽 번호. 쪽 이동 링크가 없으면 1."""
    soup = BeautifulSoup(html, "html.parser")
    numbers = [1]
    for a in soup.select(".pagination-wrap a.page-numbers"):
        m = re.search(r"/page/(\d+)", str(a.get("href", "")))
        if m:
            numbers.append(int(m.group(1)))
    return max(numbers)


def count_chars(markdown: str) -> int:
    """사람이 읽을 글자 수(공백·마크다운 기호·링크 주소·이미지/임베드 표시 제외)."""
    text = re.sub(r"\[(이미지|임베드)[^\]]*\]", "", markdown)
    text = re.sub(r"\]\(https?://[^)]*\)", "", text)
    return len(re.sub(r"[#>|\-*`\[\]()\s]|---", "", text))


def classify(chars: int, images: int, embeds: int) -> tuple[str, str]:
    """(상태, 사람이 읽을 이유). 상태: ok / short / empty."""
    if chars < EMPTY_MAX_CHARS:
        if embeds:
            return "empty", "본문 없음 — 동적 위젯(임베드)만 있음"
        if images:
            return "empty", "본문 없음 — 이미지만 있음"
        return "empty", "본문 없음 — 화면이 열린 뒤 채워지는 페이지로 보임"
    if chars < OK_MIN_CHARS:
        return "short", "본문이 짧음 — 사이트 화면과 비교해 빠진 내용이 없는지 확인"
    return "ok", ""


# ---------------------------------------------------------------- 레코드·저장

@dataclass
class PageRecord:
    slug: str
    url: str
    final_url: str = ""
    title: str = ""
    body: str = ""
    status: str = "ok"  # ok / short / empty / error / blocked / external / duplicate
    note: str = ""
    chars: int = 0
    headings: int = 0
    tables: int = 0
    images: int = 0
    content_hash: str = ""
    source: str = "web"  # web / manual
    fetched_at: str = ""
    group: str = ""
    tier: int = 1

    @property
    def writable(self) -> bool:
        return self.status in ("ok", "short") or (self.source == "manual" and self.status != "error")


def _hash(body: str) -> str:
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scrape_one(
    slug: str,
    url: str,
    *,
    fetcher: Fetcher | None = None,
    robots: robotparser.RobotFileParser | None = None,
    group: str = "",
    tier: int = 1,
    paginate: bool = False,
    delay: float = 1.0,
) -> PageRecord:
    rec = PageRecord(slug=slug, url=url, group=group, tier=tier)
    if robots is not None and not robots.can_fetch(USER_AGENT, url):
        rec.status, rec.note = "blocked", "robots.txt가 막음 — 수집하지 않음"
        return rec
    try:
        html, final_url = (fetcher or fetch_html)(url)
    except FetchError as e:
        rec.status, rec.note = "error", f"가져오지 못함: {e}"
        return rec
    rec.final_url = final_url
    if urlparse(final_url).netloc not in ("", SITE_HOST):
        rec.status, rec.note = "external", f"다른 사이트로 이동함: {final_url}"
        return rec
    page = extract_page(html, final_url)
    if paginate:
        bodies = [page.body]
        for n in range(2, min(find_last_page(html), MAX_PAGES) + 1):
            time.sleep(delay)
            try:
                more_html, _ = (fetcher or fetch_html)(f"{final_url.rstrip('/')}/page/{n}")
            except FetchError as e:
                rec.note = f"{n}쪽을 못 가져옴: {e}"
                break
            more = extract_page(more_html, final_url)
            bodies.append(more.body)
            page.tables += more.tables
            page.images += more.images
            page.headings += more.headings
            page.faq_items += more.faq_items
        page.body = "\n\n".join(b for b in bodies if b)
    rec.title, rec.body = page.title or slug, page.body
    rec.chars = count_chars(page.body)
    rec.headings, rec.tables, rec.images = len(page.headings), page.tables, page.images
    rec.status, rec.note = classify(rec.chars, page.images, len(page.embeds))
    if page.faq_items:
        rec.note = f"FAQ {page.faq_items}건" + (f" ({rec.note})" if rec.note else "")
    elif page.repeated_removed and rec.status == "ok":
        rec.note = f"같은 문장 {page.repeated_removed}줄 중복 제거"
    rec.content_hash = _hash(page.body)
    rec.fetched_at = _now()
    return rec


def load_manual(slug: str, manual_dir: Path, url: str = "") -> PageRecord | None:
    """`manual/<slug>.md`가 있으면 그 내용이 사이트 추출보다 우선한다."""
    path = manual_dir / f"{slug}.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    fm_url = ""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            m = re.search(r"^source_url:\s*\"?([^\"\n]+)\"?\s*$", text[4:end], re.MULTILINE)
            fm_url = m.group(1) if m else ""
            text = text[end + 5:]
    text = text.strip()
    title = slug
    m = re.match(r"#\s+(.+)\n?", text)
    if m:
        title = m.group(1).strip()
        text = text[m.end():].strip()
    rec = PageRecord(slug=slug, url=fm_url or url, final_url=fm_url or url, title=title, body=text)
    rec.source = "manual"
    rec.chars = count_chars(text)
    rec.status = "ok" if rec.chars >= 20 else "short"
    rec.note = "사람이 직접 넣은 내용(manual) — 사이트 추출보다 우선"
    rec.content_hash = _hash(text)
    return rec


def mark_duplicates(records: list[PageRecord]) -> None:
    """같은 내용(해시)이 앞서 나온 페이지와 같으면 duplicate로 바꾼다."""
    seen: dict[str, str] = {}
    for rec in records:
        if rec.status != "ok" or not rec.content_hash:
            continue
        first = seen.get(rec.content_hash)
        if first is None:
            seen[rec.content_hash] = rec.slug
        else:
            rec.status, rec.note = "duplicate", f"{first}과 같은 내용 — 저장하지 않음"


def render_markdown(rec: PageRecord) -> str:
    fm = [
        "---",
        f"slug: {rec.slug}",
        f"title: {json.dumps(rec.title, ensure_ascii=False)}",
        f"source_url: {json.dumps(rec.final_url or rec.url, ensure_ascii=False)}",
        f"source: {rec.source}",
    ]
    if rec.source == "web":
        fm.append(f"fetched_at: {rec.fetched_at}")
    fm += [f"content_hash: {rec.content_hash}", f"chars: {rec.chars}", f"status: {rec.status}", "---", ""]
    return "\n".join(fm) + f"# {rec.title}\n\n{rec.body}\n"


def write_record(rec: PageRecord, out_dir: Path) -> str:
    """created / updated / unchanged. 내용(해시)이 같으면 다시 쓰지 않는다."""
    path = out_dir / f"{rec.slug}.md"
    if path.exists():
        m = re.search(r"^content_hash:\s*(\S+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
        if m and m.group(1) == rec.content_hash:
            return "unchanged"
        action = "updated"
    else:
        action = "created"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rec), encoding="utf-8")
    return action


# ---------------------------------------------------------------- 대상 목록·상태·보고서

def load_sources(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data["pages"]
    slugs = [p["slug"] for p in pages]
    if len(slugs) != len(set(slugs)):
        raise ValueError("sources.json에 slug가 중복됨")
    for p in pages:
        if not str(p["path"]).startswith("/"):
            raise ValueError(f"path는 /로 시작해야 함: {p['slug']}")
    return list(pages)


def update_state(
    state_path: Path, records: list[PageRecord], keep: set[str] | None = None
) -> dict[str, dict[str, object]]:
    """_state.json: 마지막 수집 결과(시각 없음 — 바뀐 것만 diff에 보이게).
    이번에 돌린 slug만 갱신하고, keep을 주면 sources.json에서 빠진 slug는 지운다."""
    state: dict[str, dict[str, object]] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    if keep is not None:
        state = {k: v for k, v in state.items() if k in keep}
    for rec in records:
        state[rec.slug] = {
            "title": rec.title, "url": rec.final_url or rec.url, "status": rec.status,
            "note": rec.note, "chars": rec.chars, "headings": rec.headings, "tables": rec.tables,
            "images": rec.images, "source": rec.source, "group": rec.group, "tier": rec.tier,
        }
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return state


_STATUS_ORDER = ["ok", "short", "empty", "error", "blocked", "external", "duplicate"]


def render_report(state: dict[str, dict[str, object]]) -> str:
    counts: dict[str, int] = {}
    for v in state.values():
        counts[str(v["status"])] = counts.get(str(v["status"]), 0) + 1
    lines = [
        "# 홈페이지 안내 페이지 수집 보고서",
        "",
        "> `python scripts/scrape_pages.py`가 자동으로 만듭니다. 직접 고치지 마세요.",
        "",
        "요약: " + ", ".join(f"{s} {counts[s]}" for s in _STATUS_ORDER if s in counts),
        "",
        "| slug | 제목 | 상태 | 글자 수 | 제목 수 | 표 | 출처 |",
        "|---|---|---|---|---|---|---|",
    ]
    for slug in sorted(state):
        v = state[slug]
        lines.append(
            f"| {slug} | {v['title']} | {v['status']} | {v['chars']} | {v['headings']} "
            f"| {v['tables']} | {v['source']} |"
        )
    todo = [s for s in sorted(state) if state[s]["status"] in ("empty", "short", "error")
            and state[s]["source"] != "manual"]
    lines += ["", "## 사람이 확인·입력할 페이지", ""]
    if not todo:
        lines.append("없음")
    for slug in todo:
        v = state[slug]
        lines.append(
            f"- `{slug}` ({v['status']}) {v['url']} — {v['note']} → "
            f"`data/pages/manual/{slug}.md`에 복붙하거나 스크린샷을 주세요"
        )
    dups = [s for s in sorted(state) if state[s]["status"] == "duplicate"]
    if dups:
        lines += ["", "## 중복 (저장 안 함)", ""] + [f"- `{s}`: {state[s]['note']}" for s in dups]
    return "\n".join(lines) + "\n"


def readable_slug(url: str) -> str:
    return unquote(urlparse(url).path.strip("/")) or "(home)"
