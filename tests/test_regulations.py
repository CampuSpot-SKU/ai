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
