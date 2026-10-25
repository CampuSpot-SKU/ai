from __future__ import annotations

from ai.privacy import OMITTED, redact_personal_emails


def test_personal_domains_are_replaced_and_school_mail_kept() -> None:
    text = "문의: kim@naver.com, lee@gmail.com / 학교: park@skuniv.ac.kr"
    out = redact_personal_emails(text)
    assert "kim@naver.com" not in out and "lee@gmail.com" not in out
    assert out.count(OMITTED) == 2 and "park@skuniv.ac.kr" in out


def test_organization_mail_is_kept() -> None:
    text = "접수: contest@seoul.go.kr, help@nurimedia.co.kr"
    assert redact_personal_emails(text) == text


def test_mailto_markdown_link_is_replaced_whole() -> None:
    out = redact_personal_emails("[hong@daum.net](mailto:hong@daum.net) 로 보내세요")
    assert out == f"{OMITTED} 로 보내세요"
    kept = "[a@skuniv.ac.kr](mailto:a@skuniv.ac.kr)"
    assert redact_personal_emails(kept) == kept


def test_case_insensitive_and_idempotent() -> None:
    once = redact_personal_emails("문의 Kim@NAVER.com")
    assert OMITTED in once and "@" not in once
    assert redact_personal_emails(once) == once


def test_trailing_period_and_korean_neighbors() -> None:
    out = redact_personal_emails("메일은 abc@gmail.com. 감사합니다")
    assert out == f"메일은 {OMITTED}. 감사합니다"


def test_space_around_at_sign_and_typo_domain() -> None:
    assert redact_personal_emails("이메일: tnwjd8768 @hanmail.net") == f"이메일: {OMITTED}"
    assert redact_personal_emails("문의 02-1234 / surimfd@gmail.coma") == f"문의 02-1234 / {OMITTED}"
    # 기관 도메인이 개인 도메인으로 시작하더라도(me.com → me.company.kr) 건드리지 않는다
    assert redact_personal_emails("x@me.company.kr") == "x@me.company.kr"
