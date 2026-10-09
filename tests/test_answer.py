from __future__ import annotations

from typing import Any

import pytest

from ai import answer as ans
from ai import rag


def _hit(text: str, dist: float = 0.2, title: str = "학칙 제29조(휴학)", **kw: Any) -> rag.Hit:
    base: dict[str, Any] = {
        "doc_type": "학칙", "article_no": "제29조", "source_url": "https://x.kr/r#29", "published_at": None,
    }
    base.update(kw)
    return rag.Hit(text, title, base["doc_type"], base["article_no"], base["source_url"], dist, base["published_at"])


def _g(**kw: Any) -> ans.GeminiAnswer:
    return ans.GeminiAnswer(**{"answerable": True, "answer": "학칙 제29조에 따르면 휴학할 수 있어요.", "used": [1], **kw})


def test_extract_article_nos() -> None:
    assert rag.extract_article_nos("제 29 조랑 제5조의2 알려줘, 제29조도") == ["제29조", "제5조의2"]
    assert rag.extract_article_nos("휴학 기간이 궁금해요") == []


def test_select_hits_drops_far_and_near_duplicates() -> None:
    a = _hit("휴학 신청은 학기 시작 전에 합니다. " * 10, 0.2)
    dup = _hit("휴학 신청은 학기 시작 전에 합니다. " * 10, 0.25, title="휴학 안내", doc_type="안내", article_no=None)
    far = _hit("전혀 다른 이야기", 0.9)
    other = _hit("장학금은 성적에 따라 지급됩니다. " * 5, 0.3, title="장학")
    assert ans.select_hits([a, dup, far, other], 0.55) == [a, other]


def test_validate_maps_used_numbers_to_unique_sources() -> None:
    hits = [_hit("a"), _hit("b", title="안내", doc_type="안내", article_no=None, source_url=None)]
    r = ans.validate(_g(used=[2, 1, 2]), hits)
    assert [s.title for s in r.sources] == ["안내", "학칙 제29조(휴학)"]
    assert r.sources[0].article_no is None and r.sources[0].url is None


def test_validate_unanswerable_gives_notice_without_sources() -> None:
    r = ans.validate(_g(answerable=False, answer="", used=[]), [_hit("a")])
    assert r.answer == ans.NO_EVIDENCE_ANSWER and r.sources == []


@pytest.mark.parametrize(
    "g",
    [_g(used=[]), _g(used=[3]), _g(used=[0]), _g(answer=""), _g(answer="**굵게**"), _g(answer="가" * 1000)],
)
def test_validate_rejects_bad_output(g: ans.GeminiAnswer) -> None:
    with pytest.raises(ans.AnswerError):
        ans.validate(g, [_hit("a")])


def test_answer_without_evidence_skips_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any) -> ans.GeminiAnswer:
        raise AssertionError("Gemini를 부르면 안 됨")

    monkeypatch.setattr(ans, "_call_gemini", boom)
    r = ans.answer("질문", session=None, searcher=lambda q, **_k: [_hit("멀어요", 0.99)])
    assert r.answer == ans.NO_EVIDENCE_ANSWER and r.sources == []


def test_answer_retries_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(_q: str, _h: list[rag.Hit]) -> ans.GeminiAnswer:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("quota")
        return _g()

    monkeypatch.setattr(ans, "_call_gemini", flaky)
    r = ans.answer("휴학", session=None, searcher=lambda q, **_k: [_hit("휴학 안내")])
    assert calls["n"] == 2 and r.sources[0].article_no == "제29조"


def test_answer_fails_after_two_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any) -> ans.GeminiAnswer:
        raise RuntimeError("down")

    monkeypatch.setattr(ans, "_call_gemini", boom)
    with pytest.raises(ans.AnswerError):
        ans.answer("휴학", session=None, searcher=lambda q, **_k: [_hit("휴학 안내")])


def test_prompt_content_shows_notice_date_in_kst() -> None:
    notice = _hit("공지 본문", title="수강신청 안내", doc_type="공지", article_no=None, published_at="2026-10-05T16:00:00Z")
    text = ans._build_content("수강신청 언제?", [notice])
    assert "[1] 제목: 수강신청 안내 (공지, 게시일 2026-10-06)" in text and "수강신청 언제?" in text


def test_search_hybrid_puts_direct_article_first_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    vec = [_hit("벡터1", 0.2), _hit("조문", 0.3)]
    monkeypatch.setattr(rag, "search", lambda *_a, **_k: vec)
    monkeypatch.setattr(rag, "search_articles", lambda nos, **_k: [_hit("조문", 0.0)] if nos else [])
    out = rag.search_hybrid("제29조 알려줘", session=None)
    assert [h.chunk_text for h in out] == ["조문", "벡터1"]
    assert [h.chunk_text for h in rag.search_hybrid("휴학", session=None)] == ["벡터1", "조문"]
