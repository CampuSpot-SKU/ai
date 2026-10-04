"""학과 교수진 소개 페이지 수집 (1-4a).

조직도 데이터에 적힌 각 학과의 교수진 페이지(`https://<학과>.skuniv.ac.kr/<학과>_professor`)에서
교수별 직위·학위·전공분야·연구실·전화·이메일을 읽어 교수 1명당 한 덩어리의 마크다운으로 만든다.

- 퇴임교수는 "퇴임교수"라는 사실과 이름·전공만 적고 연락처(연구실·전화·이메일)는 싣지 않는다.
- 페이지에 없는 항목은 지어내지 않는다.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag

from ai import page_scraper as ps

GENERIC_SECTIONS = {"", "professors", "교수진", "교수진소개", "교수진 소개"}
RETIRED_WORDS = ("퇴임", "정년퇴직")
CONTACT_KEYS = ("연구실", "전화", "email", "e-mail", "이메일", "홈페이지", "lab", "office")
MIN_TOTAL = 150  # 이보다 적으면 사이트 구조가 바뀐 것으로 보고 저장하지 않는다.

Fetcher = Callable[[str], tuple[str, str]]


class ProfessorsError(Exception):
    """교수진 페이지를 못 받았거나 형식이 달라졌을 때."""


def _clean(text: str) -> str:
    text = text.replace("：", ":").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _phone_dashes(text: str) -> str:
    return re.sub(r"\b(0\d{1,2})[ .]+(\d{3,4})[ .]+(\d{4})\b", r"\1-\2-\3", text)


def _normalize_line(text: str) -> str:
    text = _phone_dashes(_clean(text))
    return re.sub(r"^(?:e-?mail)\s*:?\s*", "이메일: ", text, flags=re.IGNORECASE)


def _row_lines(row: Tag) -> list[str]:
    cells = row.find_all("td")
    cell = cells[-1] if cells else row
    parts = [_normalize_line(e.get_text(" ")) for e in cell.find_all(["h4", "li", "p"])]
    if not [p for p in parts if p]:
        parts = [_normalize_line(t) for t in cell.get_text("\n").split("\n")]
    lines: list[str] = []
    for part in parts:
        if part and part not in lines:
            lines.append(part)
    return lines


def _is_contact(line: str) -> bool:
    low = line.lower()
    return any(low.startswith(k) for k in CONTACT_KEYS) or "@" in line


def parse_professors(html: str) -> list[dict[str, Any]]:
    """페이지의 교수 목록. 항목: section(구분), name, lines(이름 제외 줄), retired."""
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".opage") or soup.select_one("article .entry-content")
    if container is None:
        return []
    out: list[dict[str, Any]] = []
    for row in container.select("tr"):
        lines = _row_lines(row)
        if not lines:
            continue
        panel = row.find_parent(class_="panel")
        h3 = panel.select_one("h3") if isinstance(panel, Tag) else None
        section = _clean(h3.get_text(" ")) if h3 else ""
        out.append(
            {
                "section": section,
                "name": lines[0],
                "lines": lines[1:],
                "retired": any(w in section for w in RETIRED_WORDS),
            }
        )
    return out


def collect(
    sites: list[dict[str, str]],
    *,
    fetcher: Fetcher | None = None,
    delay: float = 1.0,
) -> dict[str, Any]:
    """sites(`contacts.build_sites` 결과)에서 교수진 주소를 모아 읽는다. 하나라도 실패하면 예외."""
    groups: dict[str, list[str]] = {}
    for site in sites:
        if site.get("professors"):
            groups.setdefault(site["professors"], []).append(site["name"])
    fetch = fetcher or ps.fetch_html
    robots: dict[str, Any] = {}
    pages: dict[str, Any] = {}
    for url, names in groups.items():
        host = urlparse(url).netloc
        if fetcher is None:
            if host not in robots:
                robots[host] = ps.load_robots(f"https://{host}")
            if not robots[host].can_fetch(ps.USER_AGENT, url):
                raise ProfessorsError(f"robots.txt가 수집을 막고 있습니다: {url}")
        try:
            html, _ = fetch(url)
        except ps.FetchError as exc:
            raise ProfessorsError(f"{url} 를 받지 못했습니다: {exc}") from exc
        found = parse_professors(html)
        if not found:
            raise ProfessorsError(
                f"{url} 에서 교수를 한 명도 못 찾았습니다(페이지 구조가 바뀐 듯)."
            )
        pages[url] = {"departments": names, "professors": found}
        if fetcher is None:
            time.sleep(delay)
    total = sum(len(p["professors"]) for p in pages.values())
    if total < MIN_TOTAL:
        raise ProfessorsError(
            f"교수가 {total}명뿐입니다(기준 {MIN_TOTAL}명 이상). 저장하지 않습니다."
        )
    return {"pages": pages}


def _label(page: dict[str, Any], section: str) -> str:
    depts = page["departments"]
    generic = section.replace(" ", "").lower() in {s.replace(" ", "") for s in GENERIC_SECTIONS}
    if len(depts) > 1:
        return section if not generic else " · ".join(depts)
    dept = depts[0]
    if generic or section == dept:
        return dept
    return f"{dept} · {section}"


def render_markdown(data: dict[str, Any], collected: str) -> str:
    lines = [
        "# 학과 교수진 연락처",
        "",
        f"> 출처: 각 학과 홈페이지의 '교수진 소개' 페이지({collected} 수집). 페이지에 적힌 내용만 정리했고, 없는 항목은 비워 둠.",
        "> 퇴임교수는 퇴임했다는 사실과 이름·전공만 적고 연락처는 싣지 않음.",
    ]
    for url, page in data["pages"].items():
        lines += ["", f"## {' · '.join(page['departments'])}", "", f"- 교수진 소개 페이지: {url}"]
        for prof in page["professors"]:
            lines += ["", f"### {prof['name']} ({_label(page, prof['section'])})"]
            if prof["retired"]:
                lines.append("- 퇴임교수 (퇴임함. 연락처는 안내하지 않음)")
                lines += [f"- {ln}" for ln in prof["lines"] if not _is_contact(ln)]
            else:
                lines += [f"- {ln}" for ln in prof["lines"]]
    lines.append("")
    return "\n".join(lines)


def diff_data(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    out: list[str] = []
    old_pages = old.get("pages", {})
    for url, page in new["pages"].items():
        before = {p["name"]: p for p in old_pages.get(url, {}).get("professors", [])}
        after = {p["name"]: p for p in page["professors"]}
        dept = " · ".join(page["departments"])
        for name, prof in after.items():
            if name not in before:
                out.append(f"교수 추가: {dept} {name}" + (" (퇴임)" if prof["retired"] else ""))
            elif before[name] != prof:
                out.append(f"교수 정보 변경: {dept} {name}")
        for name in before:
            if name not in after:
                out.append(f"교수 삭제: {dept} {name}")
    for url in old_pages:
        if url not in new["pages"]:
            out.append(f"교수진 페이지 삭제: {url}")
    return out
