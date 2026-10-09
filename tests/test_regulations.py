from __future__ import annotations

from pathlib import Path

import pytest

from ai import rag
from ai import regulations as reg

SAMPLE = """\
                                                  서경대학교학칙

                                     서경대학교 학칙

제정 1981. 3. 1. 개정 2025. 10. 1

                                     제 1 장           총         칙
제 1 조(목적) 이 학칙은 서경대학교(이하 “본교”라 한다)의 목적을
  다음과 같이 정한다.(개정 2014. 4. 1)
    1. 건학이념 : 홍익인간
제 2 조의2(계약학과) ① 계약학과를 둔다.(신설 2011. 3. 1)
  ② 세부사항은 별도로 정한다.
제2편 학칙    제1장 서경대학교
                            제 8 장      휴·복학 및 퇴학
제 28 조(휴학) 병역, 질병 등의 사유로 휴학원을 제출하여야 한다.
\f                                                  서경대학교학칙

  한다.(개정 2014. 3. 1)
                         제 9 장    교과의 이수
                         제 2 절   수업
제 36 조(수업) 수업은 강의로 한다.
제 56 조의2(삭제 2025. 10. 1)

                            부    칙
1. 이 학칙은 1981년 3월 1일부터 시행한다.
제 99 조(부칙 안의 가짜 조) 이건 들어가면 안 됨
"""


def test_parse_articles_structure() -> None:
    arts = reg.parse_articles(SAMPLE)
    assert [a.article_no for a in arts] == ["제1조", "제2조의2", "제28조", "제36조"]  # 삭제 조 제외
    first = arts[0]
    assert first.title == "목적" and first.chapter == "제1장 총칙"
    assert "건학이념" in first.body and "(개정 2014. 4. 1)" in first.body
    assert arts[1].branch == 2 and "② 세부사항" in arts[1].body


def test_page_noise_and_page_break_removed() -> None:
    arts = {a.article_no: a for a in reg.parse_articles(SAMPLE)}
    body = arts["제28조"].body
    assert "서경대학교학칙" not in body and "제2편" not in body and "\f" not in body
    assert body.splitlines()[-1].startswith("한다.")  # 쪽 경계 뒤 이어지는 줄이 붙음
    assert "\n\n" not in body


def test_chapter_and_section_labels() -> None:
    arts = {a.article_no: a for a in reg.parse_articles(SAMPLE)}
    assert arts["제28조"].chapter == "제8장 휴·복학 및 퇴학" and arts["제28조"].section == ""
    assert arts["제36조"].chapter == "제9장 교과의 이수"
    assert arts["제36조"].section == "제2절 수업"


def test_body_stops_at_buchik() -> None:
    assert "제99조" not in {a.article_no for a in reg.parse_articles(SAMPLE)}


def test_pages_and_chunk_head() -> None:
    pages = reg.to_pages(reg.parse_articles(SAMPLE), "2025-10-01", "2025.10.1")
    p = next(x for x in pages if x.article_no == "제28조")
    assert p.doc_type == "학칙" and p.title == "학칙 제28조(휴학)"
    assert p.source_url == "https://www.skuniv.ac.kr/rules#학칙-제28조"
    assert p.published_at == "2025-10-01T00:00:00Z"
    chunks = rag.chunk_page(p)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("[학칙 제28조(휴학) · 제8장 휴·복학 및 퇴학 · 2025.10.1 시행 기준]")
    assert len({x.source_url for x in pages}) == len(pages)


def test_long_article_is_split_with_head_on_each_chunk() -> None:
    long = rag.Page(
        slug="x",
        title="학칙 제1조(긴 조)",
        source_url="u",
        body="\n".join(f"{'가' * 100}" for _ in range(40)),
        doc_type="학칙",
        date_label="2025.10.1",
    )
    chunks = rag.chunk_page(long)
    assert len(chunks) >= 3
    assert all(c.text.startswith("[학칙 제1조(긴 조) · 2025.10.1 시행 기준]") for c in chunks)


def test_base_date_from_filename_and_latest(tmp_path: Path) -> None:
    assert reg.base_date_from_name(Path("hakchik-20251001.txt")) == ("2025-10-01", "2025.10.1")
    with pytest.raises(ValueError):
        reg.base_date_from_name(Path("hakchik.txt"))
    (tmp_path / "hakchik-20240901.txt").write_text("x", encoding="utf-8")
    (tmp_path / "hakchik-20261001.txt").write_text("x", encoding="utf-8")
    assert reg.latest_file(tmp_path).name == "hakchik-20261001.txt"


def test_real_file_article_count() -> None:
    pages = reg.load_pages()
    assert len(pages) >= 100  # 2025.10.1판: 조 122개 중 삭제 10개 제외 = 112
    assert pages[0].article_no == "제1조" and all(p.body for p in pages)


BOOK = """\
                                   서경대학교 규정집

                                  교원 인사 규정

제정 2001. 3. 1.
개정 2020. 9. 1.



                       제 1 장    총     칙
제 1 조(목적) 이 규정은 교원 인사에 관한 사항을 정한다.
제 2 조(임용) 교원은 총장이 임용한다.
\f제3편 행정   제2장 조직및인사행정
  ② 임용 기간은 따로 정한다.
제 3 조(삭제 2020. 9. 1)
\n                          - 12 -
\f교원인사규정

부    칙
1. 이 규정은 2001년 3월 1일부터 시행한다.

                                  ∙ 계절학기 운영에 관한 시행세칙 ∙

개정 2019. 1. 1.

제 1 조(목적) 이 세칙은 계절학기 운영을 정한다.
제 1 조(중복) 표 안에 같은 번호가 또 나옴
부    칙
1. 시행한다.

                                  서경대학교 학칙

제 1 조(목적) 학칙 본문은 규정집에서 건너뛴다.

                                  구 규정

폐지 2020. 1. 1.

제 1 조(목적) 폐지된 규정
"""


def test_book_lines_drop_running_heads_and_footers() -> None:
    lines = reg.book_lines(BOOK)
    joined = "\n".join(lines)
    assert "제3편 행정" not in joined and "교원인사규정" not in joined and "- 12 -" not in joined
    assert "② 임용 기간은 따로 정한다." in joined


def test_split_and_parse_book() -> None:
    regs = reg.parse_book(BOOK)
    assert [r.name for r in regs] == ["교원 인사 규정", "계절학기 운영에 관한 시행세칙"]  # 학칙 본문·폐지 규정 제외
    first = regs[0]
    assert [a.article_no for a in first.articles] == ["제1조", "제2조"]  # 삭제 조 제외
    assert first.articles[0].chapter == "제1장 총칙"
    assert first.articles[1].body.splitlines()[-1] == "② 임용 기간은 따로 정한다."  # 쪽 경계를 넘어 이어짐
    assert "부칙" not in first.articles[1].body and "시행한다" not in first.articles[1].body


def test_book_pages_titles_urls_and_chunk_head() -> None:
    pages = reg.book_pages(reg.parse_book(BOOK), "2024-09-01", "2024.9.1 기준 규정집")
    assert [p.title for p in pages][:2] == ["교원 인사 규정 제1조(목적)", "교원 인사 규정 제2조(임용)"]
    assert pages[0].source_url == "https://www.skuniv.ac.kr/rules#규정집-교원인사규정-제1조"
    assert len({p.source_url for p in pages}) == len(pages)  # 같은 규정 안의 중복 조 번호도 구분됨
    assert pages[-1].source_url.endswith("-제1조-2")
    assert all(p.doc_type == "학칙" and p.article_no for p in pages)
    head = rag.chunk_page(pages[0])[0].text.splitlines()[0]
    assert head == "[교원 인사 규정 제1조(목적) · 제1장 총칙 · 2024.9.1 기준 규정집]"


def test_real_book_counts() -> None:
    pages = reg.load_book_pages()
    assert 3400 <= len(pages) <= 3500  # 2024.9.1 규정집: 규정 229개, 조 3,456개(삭제 41개 제외)
    assert len({p.source_url for p in pages}) == len(pages)
    assert all(p.body.strip() for p in pages)
    assert not any("서경대학교 학칙 제" in p.title for p in pages)
    assert len(reg.load_all_pages()) == len(reg.load_pages()) + len(pages)


def test_strip_revisions_keeps_deletion_and_plain_text() -> None:
    body = (
        "① 위원회의 사무는 인사과에서 관장한다.(개정 2017. 9. 1, 20. 9. 1, 24. 3. 1)\n"
        "② 임용은 총장이 한다.(신\n설 2021. 3. 1)\n"
        "③ (삭제 2020. 9. 1)\n"
        "④ 건설공사(전문공사를 제외한다)는 따로 정한다.(조 개정 2010. 3. 1)"
    )
    out = reg.strip_revisions(body)
    assert out.splitlines() == [
        "① 위원회의 사무는 인사과에서 관장한다.",
        "② 임용은 총장이 한다.",
        "③ (삭제 2020. 9. 1)",
        "④ 건설공사(전문공사를 제외한다)는 따로 정한다.",
    ]


def test_book_pages_have_no_revision_tails() -> None:
    import re

    pages = reg.load_book_pages()
    bad = [p.title for p in pages if re.search(r"\((본?조 )?(개정|신설)\s*\d", p.body)]
    assert bad == []
