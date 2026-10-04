from __future__ import annotations

import pytest

from ai import page_scraper as ps
from ai import professors as pf

HTML = """
<div class="opage"><div class="coda-slider-wrapper">
<div class="panel"><div class="panel-wrapper"><h3 class="title"><em>교수진 소개</em></h3>
<table><tbody>
<tr><td><img src="a.png"></td><td>
  <h4>김선희</h4>
  <ul><li>공학박사</li><li>전공분야 : 생체신호분석</li>
  <li>연구실 : 한림관 804호 / 02 940 7515</li>
  <li><span>Email: </span><a href="mailto:a@skuniv.ac.kr">a@skuniv.ac.kr</a></li></ul>
</td></tr>
<tr><td><img src="b.png"></td><td>
  <h4>장웅상</h4>
  <ul><li>학부장/교수</li><li>연구실 : 대일관 212호</li></ul>
</td></tr>
</tbody></table></div></div>
<div class="panel"><div class="panel-wrapper"><h3 class="title"><em>퇴임교수</em></h3>
<table><tbody>
<tr><td><img src="c.png"></td><td>
  <h4>박종준</h4>
  <ul><li>공학박사</li><li>전공분야 : 인터넷, 정보기술</li>
  <li>Email : <a href="mailto:jong@x.kr">jong@x.kr</a></li></ul>
</td></tr>
</tbody></table></div></div>
</div></div>
"""


def test_parse_sections_and_retired() -> None:
    profs = pf.parse_professors(HTML)
    assert [p["name"] for p in profs] == ["김선희", "장웅상", "박종준"]
    assert profs[0]["lines"][2] == "연구실 : 한림관 804호 / 02-940-7515"
    assert profs[0]["lines"][3] == "이메일: a@skuniv.ac.kr"
    assert profs[1]["lines"] == ["학부장/교수", "연구실 : 대일관 212호"]
    assert [p["retired"] for p in profs] == [False, False, True]


def test_render_hides_retired_contact() -> None:
    data = {
        "pages": {
            "https://cs.skuniv.ac.kr/cs_professor": {
                "departments": ["소프트웨어학과"],
                "professors": pf.parse_professors(HTML),
            }
        }
    }
    md = pf.render_markdown(data, "2026-10-05")
    assert "### 김선희 (소프트웨어학과)" in md
    assert "이메일: a@skuniv.ac.kr" in md
    assert "### 박종준 (소프트웨어학과 · 퇴임교수)" in md and "퇴임교수 (퇴임함" in md
    assert "jong@x.kr" not in md and "- 전공분야 : 인터넷, 정보기술" in md


def test_collect_fails_on_empty_page_or_small_total() -> None:
    sites = [{"name": "A학과", "professors": "https://a.skuniv.ac.kr/a_professor"}]
    with pytest.raises(pf.ProfessorsError):
        pf.collect(sites, fetcher=lambda u: ("<div class='opage'></div>", u))
    with pytest.raises(pf.ProfessorsError):
        pf.collect(sites, fetcher=lambda u: (HTML, u))  # 3명뿐 → 기준 미달

    def boom(url: str) -> tuple[str, str]:
        raise ps.FetchError("HTTP 404")

    with pytest.raises(pf.ProfessorsError):
        pf.collect(sites, fetcher=boom)


def test_diff() -> None:
    url = "https://cs.skuniv.ac.kr/cs_professor"
    old = {
        "pages": {url: {"departments": ["소프트웨어학과"], "professors": pf.parse_professors(HTML)}}
    }
    changed = pf.parse_professors(HTML.replace("804호", "805호").replace("장웅상", "신임교수"))
    new = {"pages": {url: {"departments": ["소프트웨어학과"], "professors": changed}}}
    out = pf.diff_data(old, new)
    assert "교수 정보 변경: 소프트웨어학과 김선희" in out
    assert "교수 추가: 소프트웨어학과 신임교수" in out
    assert "교수 삭제: 소프트웨어학과 장웅상" in out
    assert pf.diff_data(old, old) == []
