from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ai import rag

LONG = "가" * 300


def _page(slug: str, body: str, url: str = "https://x.kr/a") -> rag.Page:
    return rag.Page(slug=slug, title="제목", source_url=url, body=body)


def test_front_matter_and_load(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        '---\nslug: a\ntitle: "에이"\nsource_url: "https://x.kr/a"\nstatus: ok\n---\n# 에이\n\n본문\n',
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "---\nslug: b\ntitle: 비\nsource_url: https://x.kr/b\nstatus: short\n---\n본문\n",
        encoding="utf-8",
    )
    (tmp_path / "_report.md").write_text("무시", encoding="utf-8")
    pages = rag.load_pages(tmp_path)
    assert [p.slug for p in pages] == ["a"] and pages[0].title == "에이"
    assert len(rag.load_pages(tmp_path, include_short=True)) == 2


def test_assign_urls_suffix_for_shared_url() -> None:
    url = "https://x.kr/organization-phone"
    pages = [
        _page("professors", "x", url),
        _page("organization-phone", "x", url),
        _page("department-sites", "x", url),
    ]
    rag.assign_urls(pages)
    got = {p.slug: p.source_url for p in pages}
    assert got["organization-phone"] == url
    assert got["professors"] == url + "#professors"
    assert got["department-sites"] == url + "#department-sites"


def test_heading_chunks_merge_short_and_prefix() -> None:
    body = f"# 졸업\n\n## 조기졸업\n\n짧음\n\n### 자격\n\n{LONG}\n\n## 유예\n\n{LONG}\n"
    chunks = rag.chunk_page(_page("graduation", body))
    assert len(chunks) == 2  # '조기졸업'(짧음)은 '자격'과 합쳐짐
    assert chunks[0].text.startswith("[제목 > 졸업 > 조기졸업]") or "[제목 > " in chunks[0].text
    assert "짧음" in chunks[0].text and LONG in chunks[0].text
    assert chunks[1].heading.endswith("유예")
    assert [c.ordinal for c in chunks] == [0, 1]


def test_long_section_is_split() -> None:
    body = "## 긴 절\n\n" + "\n\n".join(["나" * 600] * 5)
    chunks = rag.chunk_page(_page("x", body))
    assert len(chunks) >= 3
    assert all(len(c.text) <= rag.MAX_CHUNK_CHARS + 60 for c in chunks)


def test_professors_one_chunk_each() -> None:
    body = (
        "# 학과 교수진 연락처\n\n> 출처\n\n## 소프트웨어학과\n\n- 페이지: https://x\n\n"
        "### 김선희 (소프트웨어학과)\n- 연구실 : 한림관 804호\n- 이메일: a@x.kr\n\n"
        "### 박종준 (소프트웨어학과 · 퇴임교수)\n- 퇴임교수 (퇴임함)\n"
    )
    chunks = rag.chunk_page(_page("professors", body))
    texts = [c.text for c in chunks if "김선희" in c.heading or "박종준" in c.heading]
    assert len(texts) == 2
    assert "a@x.kr" in texts[0] and "a@x.kr" not in texts[1]
    assert texts[0].startswith("[제목 > 소프트웨어학과 > 김선희 (소프트웨어학과)]")


def test_faq_one_chunk_per_question() -> None:
    body = "# 전체 FAQ\n\n### Q. 하나\n\n(일반)\n\nA. 답1\n\n### Q. 둘\n\nA. 답2\n"
    chunks = rag.chunk_page(_page("all-faq", body))
    assert len(chunks) == 2 and "답1" in chunks[0].text and "답2" in chunks[1].text


def test_normalize_unit_length() -> None:
    v = rag._normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])


class FakeSession:
    def __init__(self, existing: list[Any]) -> None:
        self.existing = existing
        self.sql: list[str] = []

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        s = str(stmt)
        self.sql.append(s)
        if s.startswith("SELECT id, source_url"):
            return iter(self.existing)
        if "RETURNING id" in s:
            return SimpleNamespace(scalar_one=lambda: "new-id")
        return SimpleNamespace()


def _fake_embed(texts: Any, task: str) -> list[list[float]]:
    return [[1.0] + [0.0] * (rag.EMBEDDING_DIM - 1) for _ in texts]


def test_sync_adds_updates_and_skips_unchanged() -> None:
    pytest.importorskip("sqlalchemy")
    same = _page("a", f"## 절\n\n{LONG}\n", "https://x.kr/a")
    changed = _page("b", f"## 절\n\n{LONG}\n", "https://x.kr/b")
    new = _page("c", f"## 절\n\n{LONG}\n", "https://x.kr/c")
    gone_row = SimpleNamespace(id="g", source_url="https://x.kr/gone", title="t", content="old")
    session = FakeSession(
        [
            SimpleNamespace(id="1", source_url=same.source_url, title=same.title, content=same.body),
            SimpleNamespace(id="2", source_url=changed.source_url, title="옛 제목", content="옛 내용"),
            gone_row,
        ]
    )
    res = rag.sync_documents(
        [same, changed, new], session=session, embedder=_fake_embed, remove_missing=True
    )
    assert (res.added, res.updated, res.unchanged, res.removed) == (1, 1, 1, 1)
    assert res.chunks == 2
    assert sum("INSERT INTO admin_faq_embeddings" in s for s in session.sql) == 2
    assert sum("DELETE FROM admin_faq_embeddings" in s for s in session.sql) == 1


def test_notice_chunks_have_head_and_short_notice_is_title_only() -> None:
    long_body = "수강신청 변경 안내입니다. " * 20
    long_page = rag.Page(
        slug="notice-1",
        title="2학기 수강신청 안내",
        source_url="https://x.kr/notice/1",
        body=long_body,
        doc_type=rag.DOC_TYPE_NOTICE,
        category="학사",
        date_label="2026-10-06",
    )
    chunks = rag.chunk_page(long_page)
    assert chunks[0].text.startswith("[공지 · 학사 · 2학기 수강신청 안내 · 2026-10-06]")
    assert not chunks[0].meta.get("short")

    short_page = rag.Page(
        slug="notice-2",
        title="현장실습학기제 설명회 개최 안내",
        source_url="https://x.kr/notice/2",
        body="[이미지]",
        doc_type=rag.DOC_TYPE_NOTICE,
        category="일반",
        date_label="2026-10-06",
    )
    short = rag.chunk_page(short_page)
    assert len(short) == 1 and short[0].meta["short"] is True
    assert "현장실습학기제 설명회 개최 안내" in short[0].text and "원문 링크" in short[0].text


def test_rerank_prefers_recent_notice_when_distances_are_close() -> None:
    from datetime import UTC, datetime

    now = datetime(2026, 10, 6, tzinfo=UTC)
    old = rag.Hit("옛", "t", "공지", None, "u1", 0.231, "2025-10-13T01:00:00Z")
    new = rag.Hit("새", "t", "공지", None, "u2", 0.234, "2026-10-06T01:00:00Z")
    assert [h.chunk_text for h in rag.rerank([old, new], 2, now)] == ["새", "옛"]


def test_rerank_keeps_much_closer_old_notice_and_ignores_non_notices() -> None:
    from datetime import UTC, datetime

    now = datetime(2026, 10, 6, tzinfo=UTC)
    old_close = rag.Hit("옛", "t", "공지", None, "u1", 0.15, "2025-10-13T01:00:00Z")
    new_far = rag.Hit("새", "t", "공지", None, "u2", 0.30, "2026-10-06T01:00:00Z")
    guide = rag.Hit("안내", "t", "안내", None, "u3", 0.20)
    out = rag.rerank([new_far, guide, old_close], 3, now)
    assert [h.chunk_text for h in out] == ["옛", "안내", "새"]
    assert guide.score == pytest.approx(0.20)


class _FakeSession:
    def __init__(self, names: list[str] | None = None) -> None:
        self.names = names or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, stmt: Any, params: Any = None) -> list[SimpleNamespace]:
        self.calls.append((str(stmt), params or {}))
        return [SimpleNamespace(name=n) for n in self.names]


def test_match_regulations_longest_and_no_overlap() -> None:
    sess = _FakeSession(["교원 인사 규정", "비전임교원 인사 규정", "복무 규정"])
    got = rag.match_regulations("비전임교원 인사 규정 제5조가 뭐야", session=sess)
    assert got == ["비전임교원 인사 규정"]  # '교원 인사 규정'은 더 긴 이름에 포함되어 제외
    assert rag.match_regulations("학칙 제5조", session=sess) == []


def test_search_articles_scope_main_vs_named_regulation() -> None:
    sess = _FakeSession()
    rag.search_articles(["제5조"], session=sess)
    sql, params = sess.calls[-1]
    assert "d.source_url LIKE :prefix" in sql and params["prefix"].endswith("#학칙-%")
    rag.search_articles(["제5조"], session=sess, regulations=["복무 규정"])
    sql, params = sess.calls[-1]
    assert "d.title LIKE ANY(:pats)" in sql and params["pats"] == ["복무 규정 제%"]


def test_sync_embeds_and_commits_in_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sqlalchemy")
    monkeypatch.setattr(rag, "SYNC_GROUP", 1)  # 문서 1개(청크 1개 이상)마다 한 묶음
    commits: list[int] = []
    calls: list[int] = []

    class S(FakeSession):
        def commit(self) -> None:
            commits.append(1)

    def emb(texts: Any, task: str) -> list[list[float]]:
        calls.append(len(texts))
        return _fake_embed(texts, task)

    pages = [_page(n, f"## 절\n\n{LONG}\n", f"https://x.kr/{n}") for n in "abc"]
    res = rag.sync_documents(pages, session=S([]), embedder=emb)
    assert res.added == 3 and len(calls) == 3 and len(commits) == 3


def test_embed_retry_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)

    class Err(Exception):
        code = 429

    class Client:
        n = 0

        class models:
            @staticmethod
            def embed_content(**_k: Any) -> str:
                Client.n += 1
                if Client.n < 3:
                    raise Err()
                return "ok"

    assert rag._embed_with_retry(Client, "m", ["x"], None) == "ok" and Client.n == 3

    class Bad(Exception):
        code = 400

    class Client2:
        class models:
            @staticmethod
            def embed_content(**_k: Any) -> str:
                raise Bad()

    with pytest.raises(Bad):
        rag._embed_with_retry(Client2, "m", ["x"], None)
