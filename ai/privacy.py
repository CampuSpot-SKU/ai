"""개인 메일 주소 걸러내기 (S-10, 감점 요소 7 대응).

학교 홈페이지(교수진 소개·공지)에는 교수·외부 담당자의 개인 메일(naver·gmail 등)이 공개돼 있다.
공개 레포(ai)에 그대로 모아 두면 개인정보를 모아 공개하는 모양이 되므로, 수집 단계에서 개인 도메인
메일 주소를 `OMITTED` 문구로 바꾼다. 학교 도메인(skuniv.ac.kr)·기관·단체 메일은 그대로 둔다.

- 개인 도메인은 `PERSONAL_EMAIL_DOMAINS`에 적힌 포털·무료 메일 서비스만 본다(기관 메일은 손대지 않음).
- 여러 번 적용해도 결과가 같다(`OMITTED`에는 `@`가 없다).
"""

from __future__ import annotations

import re

OMITTED = "(개인 메일은 안내하지 않음)"

PERSONAL_EMAIL_DOMAINS = frozenset(
    {
        "naver.com",
        "hanmail.net",
        "daum.net",
        "gmail.com",
        "nate.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "msn.com",
        "kakao.com",
        "yahoo.com",
        "yahoo.co.kr",
        "icloud.com",
        "me.com",
        "korea.com",
        "hanmir.com",
        "lycos.co.kr",
        "paran.com",
        "empal.com",
        "protonmail.com",
    }
)

# 사이트에 `abc @hanmail.net`처럼 @ 앞뒤에 공백이 낀 채 적힌 주소도 있다.
_EMAIL = r"[A-Za-z0-9._%+\-]+[ \t]?@[ \t]?([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)"
_EMAIL_RE = re.compile(_EMAIL)
# 마크다운 메일 링크 `[글자](mailto:주소)` — 글자와 주소를 한꺼번에 지워야 빈 링크가 남지 않는다.
_MAILTO_LINK_RE = re.compile(r"\[[^\]]*\]\(mailto:([^)\s]*)\)", re.IGNORECASE)


def is_personal_domain(domain: str) -> bool:
    """정확히 같거나, 긴 도메인 뒤에 글자가 더 붙은 오타(`gmail.coma`)면 개인 도메인으로 본다."""
    d = domain.strip().lower()
    return any(d == p or (len(p) >= 9 and d.startswith(p)) for p in PERSONAL_EMAIL_DOMAINS)


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[-1] if "@" in address else ""


def redact_personal_emails(text: str) -> str:
    """개인 도메인 메일 주소(와 그 mailto 링크)를 `OMITTED`로 바꾼다. 학교·기관 메일은 그대로."""

    def mailto(m: re.Match[str]) -> str:
        return OMITTED if is_personal_domain(_domain_of(m.group(1))) else m.group(0)

    def plain(m: re.Match[str]) -> str:
        return OMITTED if is_personal_domain(m.group(1)) else m.group(0)

    return _EMAIL_RE.sub(plain, _MAILTO_LINK_RE.sub(mailto, text))
