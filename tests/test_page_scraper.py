"""홈페이지 안내 페이지 수집 도구 유닛테스트 — 네트워크 없이 저장해 둔 실제 페이지 HTML(fixtures)로 확인 (CI용)."""
import json
from pathlib import Path
from urllib import robotparser

import pytest
import scrape_pages

from ai import page_scraper as ps

FIXTURES = Path(__file__).parent / "fixtures" / "pages"
SOURCES = Path(__file__).resolve().parent.parent / "data" / "pages" / "sources.json"


def html(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


def extract(name: str) -> ps.Extracted:
    return ps.extract_page(html(name), f"https://www.skuniv.ac.kr/{name}")


def test_extract_keeps_only_main_content():
    page = extract("leave")
    assert page.title == "휴학"
    assert "## 휴학 절차" in page.body
    assert "병역 또는 질병, 출산에 의한 경우" in page.body
    # 메뉴·푸터 같은 반복 요소는 들어오지 않는다
    assert "수업·성적" not in page.body
    assert "성북구" not in page.body


def test_sidebar_menu_inside_content_is_removed():
    page = extract("course-registration")
    assert "마이크로디그리전공" not in page.body
    assert "학부 학사일정" not in page.body
    assert "학기당 12~19학점" in page.body


def test_headings_start_at_level_2_and_title_line_removed():
    page = extract("course-registration")
    assert page.body.startswith("[수강신청 바로가기]")  # 맨 위 제목 중복 줄은 제거됨
    assert "\n## 수강신청\n" in "\n" + page.body
    assert "\n### 수강신청 기간\n" in "\n" + page.body
    assert not any(line.startswith("# ") for line in page.body.split("\n"))


def test_table_becomes_markdown_with_all_rows():
    page = extract("course-registration")
    assert "| 수강 학점 (졸업논문 포함) | 등록금 납부 기준 |" in page.body
    assert "| 7~9학점 | 정규 등록금의 1/2 |" in page.body
    assert page.tables == 1


def test_rowspan_and_colspan_are_expanded_so_each_row_is_self_contained():
    page = extract("library")
    assert page.tables == 7
    # '8층'이 rowspan으로 묶여 있어도 행마다 층 이름이 따로 있어야 질문에 답할 수 있다
    assert "| 8층 | 스터디룸 | 스터디룸 | 09:00 – 20:00 | 휴관 | 10:00 – 15:00 | 휴관 |" in page.body
    assert "| 9층 | 자유열람실 | Red Room |" in page.body


def test_repeated_lines_are_dropped_but_tables_are_kept():
    line = "직전학기에 15학점 미만 이수한 경우 성적장학금 및 포상 대상에서 제외되며 학사경고를 받은 학생은 최대 15학점까지만 신청할 수 있다고 안내합니다."
    body = f"• {line}\n\n- {line}\n\n| a | b |\n| {line} | x |\n| {line} | x |"
    text, removed = ps.drop_repeated_lines(body)
    assert removed == 1
    assert text.count(line) == 3  # 첫 줄 1번 + 표 2번(표는 건드리지 않음)


def test_count_and_classify():
    assert ps.count_chars("## 제목\n\n[이미지: 포스터]\n\n[링크](https://a.b/c) 본문") == len("제목링크본문")
    assert ps.classify(500, 0, 0) == ("ok", "")
    assert ps.classify(100, 0, 0)[0] == "short"
    assert "이미지" in ps.classify(0, 2, 0)[1]
    assert "임베드" in ps.classify(0, 0, 1)[1]
    assert ps.classify(0, 0, 0)[0] == "empty"


def fake_fetcher(pages: dict[str, tuple[str, str]]):
    def fetch(url: str) -> tuple[str, str]:
        if url not in pages:
            raise ps.FetchError("HTTP 404")
        return pages[url]

    return fetch


def test_scrape_one_ok_empty_error_external_blocked():
    base = "https://www.skuniv.ac.kr"
    empty = "<html><head><title>달력 - 서경대학교</title></head><body><main><article>" \
            "<div class='entry-content'><iframe src='https://x.y/cal'></iframe></div></article></main></body></html>"
    fetch = fake_fetcher({
        f"{base}/leave": (html("leave"), f"{base}/leave"),
        f"{base}/cal": (empty, f"{base}/cal"),
        f"{base}/go": ("<html></html>", "https://other.example.com/"),
    })
    ok = ps.scrape_one("leave", f"{base}/leave", fetcher=fetch)
    assert ok.status == "ok" and ok.title == "휴학" and ok.content_hash.startswith("sha256:")
    cal = ps.scrape_one("cal", f"{base}/cal", fetcher=fetch)
    assert cal.status == "empty" and "임베드" in cal.note
    assert ps.scrape_one("nope", f"{base}/nope", fetcher=fetch).status == "error"
    assert ps.scrape_one("go", f"{base}/go", fetcher=fetch).status == "external"
    rp = robotparser.RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /leave"])
    assert ps.scrape_one("leave", f"{base}/leave", fetcher=fetch, robots=rp).status == "blocked"


def test_manual_file_overrides_and_is_written(tmp_path: Path):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "academic-calendar.md").write_text(
        "# 학부 학사일정\n\n10월 5일 개천절 대체공휴일 입니다 그리고 다른 일정도 많이 있습니다 정말입니다.\n",
        encoding="utf-8",
    )
    rec = ps.load_manual("academic-calendar", manual, "https://www.skuniv.ac.kr/academic-calendar")
    assert rec is not None and rec.source == "manual" and rec.title == "학부 학사일정"
    assert rec.writable and rec.status == "ok"
    assert ps.load_manual("none", manual) is None
    assert ps.write_record(rec, tmp_path) == "created"
    assert "source: manual" in (tmp_path / "academic-calendar.md").read_text(encoding="utf-8")


def test_write_record_is_idempotent_and_detects_changes(tmp_path: Path):
    rec = ps.scrape_one("leave", "https://www.skuniv.ac.kr/leave",
                        fetcher=fake_fetcher({"https://www.skuniv.ac.kr/leave": (html("leave"), "https://www.skuniv.ac.kr/leave")}))
    assert ps.write_record(rec, tmp_path) == "created"
    first = (tmp_path / "leave.md").read_text(encoding="utf-8")
    rec.fetched_at = "2099-01-01T00:00:00Z"  # 시각만 달라도 내용이 같으면 파일을 안 건드린다
    assert ps.write_record(rec, tmp_path) == "unchanged"
    assert (tmp_path / "leave.md").read_text(encoding="utf-8") == first
    rec.body += "\n\n새 문장"
    rec.content_hash = ps._hash(rec.body)
    assert ps.write_record(rec, tmp_path) == "updated"
    assert first.startswith("---\nslug: leave\n") and "# 휴학" in first


def test_mark_duplicates():
    a = ps.PageRecord(slug="a", url="u", body="x", content_hash="sha256:1")
    b = ps.PageRecord(slug="b", url="u", body="x", content_hash="sha256:1")
    c = ps.PageRecord(slug="c", url="u", body="y", content_hash="sha256:2")
    ps.mark_duplicates([a, b, c])
    assert (a.status, b.status, c.status) == ("ok", "duplicate", "ok")
    assert "a" in b.note and not b.writable


def test_sources_json_is_valid():
    sources = ps.load_sources(SOURCES)
    assert len(sources) >= 40
    assert {s["tier"] for s in sources} >= {1, 2}
    slugs = [s["slug"] for s in sources]
    assert len(slugs) == len(set(slugs))
    assert "academic-calendar" in slugs  # 동적 페이지도 목록에 두고 상태로 드러낸다


def test_report_lists_pages_that_need_a_human(tmp_path: Path):
    rec = ps.PageRecord(slug="organization-phone", url="https://www.skuniv.ac.kr/organization-phone",
                        title="조직도", status="empty", note="본문 없음", group="조직")
    state = ps.update_state(tmp_path / "_state.json", [rec])
    report = ps.render_report(state)
    assert "사람이 확인·입력할 페이지" in report
    assert "data/pages/manual/organization-phone.md" in report


def test_cli_end_to_end_with_fake_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    base = "https://www.skuniv.ac.kr"
    (tmp_path / "sources.json").write_text(json.dumps({"pages": [
        {"slug": "leave", "path": "/leave", "group": "학적", "tier": 1},
        {"slug": "course-registration", "path": "/course-registration", "group": "수업", "tier": 1},
        {"slug": "gone", "path": "/gone", "group": "?", "tier": 1},
    ]}), encoding="utf-8")
    pages = {
        f"{base}/robots.txt": ("User-agent: *\nDisallow:\n", f"{base}/robots.txt"),
        f"{base}/leave": (html("leave"), f"{base}/leave"),
        f"{base}/course-registration": (html("course-registration"), f"{base}/course-registration"),
    }
    monkeypatch.setattr(ps, "fetch_html", fake_fetcher(pages))
    monkeypatch.setattr(scrape_pages, "PAGES_DIR", tmp_path)
    assert scrape_pages.main(["--delay", "0"]) == 0
    assert (tmp_path / "leave.md").exists() and (tmp_path / "course-registration.md").exists()
    assert not (tmp_path / "gone.md").exists()
    report = (tmp_path / "_report.md").read_text(encoding="utf-8")
    assert "| gone |" in report and "error" in report
    capsys.readouterr()
    assert scrape_pages.main(["--delay", "0"]) == 0  # 두 번째는 파일 변경 없음
    assert "{'unchanged': 2}" in capsys.readouterr().out


def test_card_link_keeps_its_url():
    card = (
        "<html><head><title>교양영어 - 서경대학교</title></head><body><main><article><div class='entry-content'>"
        "<h3>교양영어</h3><a href='https://example.org/hackers'><div><p>Hackers</p><p>대상자 : 신입생</p></div></a>"
        "</div></article></main></body></html>"
    )
    body = ps.extract_page(card, "https://www.skuniv.ac.kr/general-english").body
    assert "Hackers" in body and "링크: https://example.org/hackers" in body


def test_short_pages_are_not_marked_duplicate():
    a = ps.PageRecord(slug="a", url="u", status="short", content_hash="sha256:1")
    b = ps.PageRecord(slug="b", url="u", status="short", content_hash="sha256:1")
    ps.mark_duplicates([a, b])
    assert (a.status, b.status) == ("short", "short")  # 사람이 확인할 페이지가 duplicate로 숨지 않게


def test_state_drops_slugs_removed_from_sources(tmp_path: Path):
    path = tmp_path / "_state.json"
    ps.update_state(path, [ps.PageRecord(slug="old", url="u"), ps.PageRecord(slug="keep", url="u")])
    state = ps.update_state(path, [ps.PageRecord(slug="keep", url="u")], keep={"keep"})
    assert set(state) == {"keep"}


def test_faq_accordion_becomes_question_sections():
    page = extract("faq")
    assert page.faq_items == 2
    assert "### Q. 학적 변동 안내\n\n(교무학적 · 교무처 · 2025-11-18)\n\nA. **휴학 절차는 어떻게 되나요?**" in page.body
    assert "**휴학 기간은 어떻게 되나요?**" in page.body  # 답변 안의 제목은 굵은 글씨(질문 제목과 안 섞임)
    assert "[서경포탈](https://sportal.skuniv.ac.kr/login)" in page.body
    assert "### Q. 웹메일 계정 안내" in page.body
    assert "FAQ 메뉴" not in page.body and "Q\n" not in page.body.replace("### Q.", "")
    # 항목마다 같은 안내 문구가 있어도 FAQ는 중복 제거를 하지 않는다
    assert page.body.count("문의는 교무처로 해주시기 바랍니다") == 6


def test_find_last_page():
    assert ps.find_last_page(html("faq")) == 3
    assert ps.find_last_page(html("leave")) == 1


def test_scrape_one_follows_pagination():
    base = "https://www.skuniv.ac.kr/all-faq"
    page1 = html("faq")
    page2 = page1.replace("학적 변동 안내", "수강 변경 안내").replace("웹메일 계정 안내", "성적 확인 안내")
    page3 = page1.replace("학적 변동 안내", "졸업 요건 안내").replace("웹메일 계정 안내", "장학금 안내")
    fetch = fake_fetcher({base: (page1, base), f"{base}/page/2": (page2, base), f"{base}/page/3": (page3, base)})
    rec = ps.scrape_one("all-faq", base, fetcher=fetch, paginate=True, delay=0)
    assert rec.body.count("### Q.") == 6
    assert "### Q. 장학금 안내" in rec.body and rec.note.startswith("FAQ 6건")
    one = ps.scrape_one("all-faq", base, fetcher=fetch, paginate=False)
    assert one.body.count("### Q.") == 2
