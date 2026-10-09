from __future__ import annotations

from datetime import UTC, datetime
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
    monkeypatch.setattr(rag, "match_regulations", lambda *_a, **_k: [])
    out = rag.search_hybrid("제29조 알려줘", session=None)
    assert [h.chunk_text for h in out] == ["조문", "벡터1"]
    assert [h.chunk_text for h in rag.search_hybrid("휴학", session=None)] == ["벡터1", "조문"]


def test_source_as_of_only_for_regulations() -> None:
    reg_hit = _hit("조문", 0.1, title="교원 인사 규정 제3조(임용)", doc_type="학칙", article_no="제3조")
    reg_hit.published_at = "2024-09-01T00:00:00Z"
    assert ans._source(reg_hit).as_of == "2024.9.1 기준"
    notice = _hit("공지", 0.1, title="공지", doc_type="공지", article_no=None, published_at="2026-10-05T16:00:00Z")
    assert ans._source(notice).as_of is None


# ---------------------------------------------------------------- 규정 개정 공지 연결 (1-4f)


def _notice(title: str, day: str) -> ans.RevisionNotice:
    y, m, d = (int(x) for x in day.split("-"))
    return ans.RevisionNotice(title, f"https://x.kr/n/{day}", datetime(y, m, d, 1, 0, tzinfo=UTC))


REG = _hit("본문", published_at="2025-10-01T00:00:00+00:00")


def test_regulation_name() -> None:
    assert ans.regulation_name("학칙 제29조(휴학)") == "학칙"
    assert ans.regulation_name("학생생활규정 제3조(상벌)") == "학생생활규정"


def test_pick_revision_only_newer_than_regulation_date() -> None:
    old = _notice("학칙 일부개정안 사전 공고", "2025-09-01")  # 규정 기준일(10/1)보다 이전 → 이미 반영된 개정
    new = _notice("학칙 일부개정안 사전 공고", "2026-01-19")
    newer = _notice("서경대학교 학칙 및 시행세칙 개정안 사전공지", "2026-09-21")
    assert ans.pick_revision([REG], [old]) is None
    assert ans.pick_revision([REG], [newer, new, old]) == newer


def test_graduate_school_rule_notice_is_not_linked_to_undergrad_rule() -> None:
    grad = _notice("대학원 학칙 일부개정 공포", "2025-10-30")
    both = _notice("대학·대학원 학칙 및 학칙 시행세칙 일부 개정안 사전 공고", "2026-04-20")
    assert ans.pick_revision([REG], [grad]) is None
    assert ans.pick_revision([REG], [grad, both]) == both


def test_non_regulation_or_undated_hits_are_ignored() -> None:
    notice = _notice("학칙 일부개정안 사전 공고", "2026-01-19")
    guide = _hit("안내", title="안내", doc_type="안내", article_no=None, published_at="2025-10-01T00:00:00+00:00")
    undated = _hit("본문", published_at=None)
    assert ans.pick_revision([guide, undated], [notice]) is None


def test_with_revision_notice_adds_sentence_and_source() -> None:
    notice = _notice("학칙 일부개정안 사전 공고", "2026-01-19")
    base = ans.AnswerResult(answer="휴학할 수 있어요.", sources=[ans._source(REG)])
    r = ans.with_revision_notice(base, [REG], [notice])
    assert r.answer.startswith("휴학할 수 있어요.") and "1월 19일에 개정 공지가 있었어요" in r.answer
    assert r.sources[-1].title == notice.title and r.sources[-1].url == notice.url
    assert r.sources[-1].as_of == "2026.1.19 게시"
    assert ans.with_revision_notice(base, [REG], []) == base


def test_answer_adds_revision_and_survives_loader_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = _hit("휴학 안내 " * 20, published_at="2025-10-01T00:00:00+00:00")
    monkeypatch.setattr(ans, "_call_gemini", lambda q, h: _g())
    notice = _notice("학칙 일부개정안 사전 공고", "2026-01-19")
    ok = ans.answer("휴학", session=object(), searcher=lambda q, **_k: [hit], revisions=lambda s: [notice])
    assert "개정 공지가 있었어요" in ok.answer and len(ok.sources) == 2

    def boom(_s: Any) -> list[ans.RevisionNotice]:
        raise RuntimeError("db down")

    bad = ans.answer("휴학", session=object(), searcher=lambda q, **_k: [hit], revisions=boom)
    assert "개정 공지" not in bad.answer and len(bad.sources) == 1
