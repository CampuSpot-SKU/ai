"""서경대 조직도·부서 연락처 수집 (1-4a).

'조직도·전화번호' 페이지는 본문이 비어 있고, 화면이 열린 뒤 사이트 안의 공개 데이터 주소
(`/wp-json/cache/api/v1/contacts/<시트 ID>/<시트 이름>`)에서 연락처·조직도를 불러와 그린다.
여기서는 그 주소를 직접 읽어 부서별 연락처 문서(마크다운)를 만들고, 이전 수집본과 비교한다.

개인 성명·이메일은 저장하지 않는다(부서·직위·담당업무·전화만).
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any
from urllib.parse import quote

from ai.page_scraper import USER_AGENT

API_URL = "https://www.skuniv.ac.kr/wp-json/cache/api/v1/contacts/{sheet_id}/{name}"
# 학교 홈페이지 자바스크립트(script.min.js)에 그대로 적혀 있는 공개 시트 ID
CONTACTS_SHEET = ("1o1ogpBtJ9GnnDhn7LjFMX0c64f42tRd5ICeckM1bk9o", "연락처")
ORG_SHEET = ("1LdqwjAgTKiNJ-Wmtz_GVdMgFIMMRMtahvHjoGUZEZaI", "조직도")

PHONE_PREFIX = "02-940-"
NO_EXT = "내선 미공개"
# 이보다 부서가 적게 오면 사이트 구조가 바뀐 것으로 보고 저장하지 않는다.
MIN_DEPARTMENTS = 30

GRAD_HOME = "https://grad.skuniv.ac.kr/"
GRAD_OFFICE = "대학원 교학과"

HOW_TO_CALL = """## 전화 거는 방법 (서경대학교)

- 대표 전화번호: 02-940-7114 (학교 홈페이지 하단 안내)
- 상황실 전화번호: 02-940-7047 (긴급상황 발생 시)
- 학교 고유 국번은 940임. 외부(휴대폰·집 전화 등)에서 부서로 직접 걸 때는 `02-940-` 뒤에 내선번호 4자리를 붙임. 예: 내선 7071 → 02-940-7071
- 교내 전화기에서는 내선번호 4자리만 누르면 됨 (일반적인 대학 내선 이용 방식이며, 서경대 전용 안내문은 확인되지 않음)
- 이 문서에 내선이 없는 부서(내선 미공개)나 학과는 대표 전화번호 02-940-7114로 걸어 부서명을 말해 연결을 요청하면 됨"""


class ContactsError(Exception):
    """연락처 데이터를 못 받았거나 형식이 달라졌을 때."""


def fetch_sheet(sheet: tuple[str, str], *, timeout: float = 20.0) -> list[dict[str, Any]]:
    url = API_URL.format(sheet_id=sheet[0], name=quote(sheet[1]))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ContactsError(f"{sheet[1]} 시트를 받지 못했습니다: {exc}") from exc
    if not isinstance(data, list):
        raise ContactsError(f"{sheet[1]} 시트 형식이 달라졌습니다 (목록이 아님)")
    return data


def _text(value: Any) -> str:
    return str(value or "").strip()


def build_snapshot(contacts: list[dict[str, Any]], org: list[dict[str, Any]]) -> dict[str, Any]:
    """부서별로 묶은 비교용 데이터. 개인 성명·이메일은 넣지 않는다."""
    paths: dict[str, list[str]] = {}
    homes: dict[str, str] = {}
    org_names: set[str] = set()
    for item in org:
        name = _text(item.get("세분류")) or _text(item.get("소분류"))
        if not name:
            continue
        org_names.add(name)
        if name not in paths:
            top, mid, sub = (
                _text(item.get("대분류")),
                _text(item.get("중분류")),
                _text(item.get("소분류")),
            )
            paths[name] = [top, mid, sub] if _text(item.get("세분류")) else [top, mid]
        if _text(item.get("홈페이지")):
            homes[name] = _text(item.get("홈페이지"))

    departments: dict[str, dict[str, Any]] = {}
    for row in contacts:
        dept = _text(row.get("부서"))
        if not dept:
            continue
        entry = departments.setdefault(
            dept, {"path": paths.get(dept, ["기타"]), "home": homes.get(dept, ""), "rows": []}
        )
        entry["rows"].append(
            {
                "직위": _text(row.get("직위")),
                "담당업무": " / ".join(
                    ln.strip() for ln in _text(row.get("담당업무")).splitlines() if ln.strip()
                ),
                "내선": _text(row.get("내선번호")),
            }
        )
    no_contact = sorted(org_names - set(departments))
    return {
        "departments": departments,
        "no_contact": no_contact,
        "sites": build_sites(org),
        "grad_ext": sorted({r["내선"] for r in departments.get(GRAD_OFFICE, {}).get("rows", [])}),
    }


def build_sites(org: list[dict[str, Any]]) -> list[dict[str, str]]:
    """학과·학부의 홈페이지와 교수진 페이지 주소. 홈페이지가 없는 학과도 빠뜨리지 않고 적는다."""
    sites: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in org:
        if _text(item.get("중분류")) != "대학":
            continue
        college = _text(item.get("소분류"))
        name = _text(item.get("세분류")) or college
        if not name or name in seen:
            continue
        seen.add(name)
        url = _text(item.get("홈페이지")).split("#")[0]
        home, professors = url, ""
        match = re.match(r"(https?://[^/]+)/[^/]+_professor/?$", url)
        if match:
            home, professors = match.group(1) + "/", url
        sites.append({"college": college, "name": name, "home": home, "professors": professors})
    return sites


def check_snapshot(snapshot: dict[str, Any]) -> None:
    count = len(snapshot["departments"])
    if count < MIN_DEPARTMENTS:
        raise ContactsError(
            f"부서가 {count}곳뿐입니다(기준 {MIN_DEPARTMENTS}곳 이상). 사이트 구조가 바뀐 것 같아 저장하지 않습니다."
        )


def _phone(ext: str) -> str:
    if not ext:
        return NO_EXT
    return PHONE_PREFIX + ext if re.fullmatch(r"\d{4}", ext) else ext


def render_markdown(snapshot: dict[str, Any], collected: str) -> str:
    lines = [
        "# 서경대학교 조직도·부서 연락처",
        "",
        f"> 출처: 서경대학교 홈페이지 '조직도·전화번호' 페이지가 불러오는 공개 연락처·조직도 데이터({collected} 수집).",
        (
            "> 부서별 전화는 `02-940-` 뒤에 내선번호 4자리를 붙인 형태임(예: 내선 7071 → 02-940-7071). "
            "개인 성명·이메일은 저장하지 않고 부서·직위·담당업무·전화만 정리함."
        ),
        "",
        HOW_TO_CALL,
    ]
    current = None
    for dept, info in snapshot["departments"].items():
        path = info["path"]
        top = " > ".join(path[:2])
        if top != current:
            current = top
            lines += ["", f"## {top}", ""]
        parent = f" (소속: {path[2]})" if len(path) > 2 and path[2] != dept else ""
        lines.append(f"### {dept}{parent}")
        if info["home"]:
            lines.append(f"- 홈페이지: {info['home']}")
        for row in info["rows"]:
            line = f"- {row['직위'] or '담당'}: {_phone(row['내선'])}"
            if row["담당업무"]:
                line += f" — {row['담당업무']}"
            lines.append(line)
        lines.append("")
    lines += [
        "",
        "## 연락처가 공개되지 않은 조직 (학과·센터 등)",
        "",
        ", ".join(snapshot["no_contact"]),
        "",
    ]
    return "\n".join(lines)


def render_sites_markdown(snapshot: dict[str, Any], collected: str) -> str:
    lines = [
        "# 학과·학부 홈페이지와 교수진 페이지 안내",
        "",
        f"> 출처: 서경대학교 홈페이지 '조직도·전화번호' 페이지의 조직도 데이터({collected} 수집).",
        (
            "> 학과별 교수 연구실·전화·이메일은 각 학과 홈페이지의 '교수진 소개' 페이지에 공개되어 있음(학과마다 공개 항목이 다름). "
            "연락처를 묻는 질문에는 가능한 한 해당 학과의 교수진 페이지 주소를 함께 안내함."
        ),
    ]
    current = None
    for site in snapshot["sites"]:
        if site["college"] != current:
            current = site["college"]
            lines += ["", f"## {current}", ""]
        if not site["home"]:
            lines.append(
                f"- {site['name']}: 학과 홈페이지가 조직도에 등록되어 있지 않음. "
                "학과 문의는 대표 전화번호 02-940-7114로 걸어 학과명을 말하고 연결을 요청"
            )
            continue
        line = f"- {site['name']}: 홈페이지 {site['home']}"
        if site["professors"]:
            line += f" / 교수진 소개 페이지 {site['professors']}"
        lines.append(line)
    ext = ", ".join(
        PHONE_PREFIX + e if re.fullmatch(r"\d{4}", e) else e for e in snapshot["grad_ext"]
    )
    lines += [
        "",
        "## 대학원",
        "",
        (
            f"- 대학원(일반대학원·미용예술대학원·실용음악대학원·융합대학원·국제융합대학원 등) 문의는 대학원 홈페이지 {GRAD_HOME} 로 안내함. "
            "학과별 세부 정보도 이 홈페이지에서 찾을 수 있음."
        ),
    ]
    if ext:
        lines.append(f"- 대학원 교학과 전화: {ext}")
    lines.append("")
    return "\n".join(lines)


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """이전/이번 수집본의 차이를 사람이 읽을 문장 목록으로."""
    out: list[str] = []
    old_d, new_d = old.get("departments", {}), new["departments"]
    for dept in new_d:
        if dept not in old_d:
            out.append(f"부서 추가: {dept}")
    for dept in old_d:
        if dept not in new_d:
            out.append(f"부서 삭제: {dept}")
    for dept, info in new_d.items():
        before = old_d.get(dept)
        if before is None:
            continue
        if before["rows"] != info["rows"]:
            old_ext = sorted(r["내선"] for r in before["rows"])
            new_ext = sorted(r["내선"] for r in info["rows"])
            if old_ext != new_ext:
                out.append(f"내선 변경: {dept} ({', '.join(old_ext)} → {', '.join(new_ext)})")
            else:
                out.append(f"직위·담당업무 변경: {dept}")
        elif before["path"] != info["path"] or before["home"] != info["home"]:
            out.append(f"소속·홈페이지 변경: {dept}")
    if old.get("no_contact") != new["no_contact"]:
        out.append("연락처 없는 조직 목록 변경")
    old_s = {x["name"]: x for x in old.get("sites", [])}
    new_s = {x["name"]: x for x in new["sites"]}
    for name, site in new_s.items():
        if name not in old_s:
            out.append(f"학과 사이트 추가: {name}")
        elif old_s[name] != site:
            out.append(f"학과 사이트 변경: {name}")
    for name in old_s:
        if name not in new_s:
            out.append(f"학과 사이트 삭제: {name}")
    if old.get("grad_ext", new["grad_ext"]) != new["grad_ext"]:
        out.append("대학원 교학과 내선 변경")
    return out
