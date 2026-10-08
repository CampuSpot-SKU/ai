"""학칙 파서 (1-4b) — 학칙 텍스트(PDF에서 뽑은 .txt)를 조(제N조) 단위 문서로 나눈다.

입력: data/regulations/hakchik-YYYYMMDD.txt (파일명 날짜 = 시행일 = 답변에 밝히는 '기준일')
출력: rag.Page 목록 — 조 1개 = 문서 1개(doc_type=학칙, article_no=제N조 / 제N조의2).
      `rag.sync_documents`로 적재하면 바뀐 조만 다시 임베딩된다.

규칙
- 본문은 첫 `제 1 조(` 부터 첫 `부 칙` 앞까지. 부칙·별표·서식은 넣지 않는다(개정 이력은 조문 끝 괄호에 이미 있음).
- 쪽 머리글·바닥글(`서경대학교학칙`, `제2편 학칙 …`)과 쪽 구분 문자를 지운다.
- 장·절 제목을 청크 머리말에 붙인다: `[학칙 제N조(제목) · 제X장 … · 제Y절 … · 2025.10.1 시행 기준]`.
- `제N조의2` 같은 가지 조문은 별도 조로 취급한다. `(삭제 …)` 조문은 건너뛴다.
- 주소: https://www.skuniv.ac.kr/rules#학칙-제N조  (학교 규정 페이지 + 조 표시. DB source_url 구분용)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ai import rag

REGULATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "regulations"
RULES_URL = "https://www.skuniv.ac.kr/rules"
DOC_TYPE_REG = rag.DOC_TYPE_REGULATION

_ARTICLE = re.compile(r"^제 *(\d+) *조(?: *의 *(\d+))? *\(([^)]*)\)\s*(.*)$")
_CHAPTER = re.compile(r"^제 *(\d+) *([장절]) +(.+)$")
_NOISE = re.compile(r"^\s*(서경대학교\s*학칙|제\s*2편\s*학칙.*)\s*$")
_BODY_END = re.compile(r"^\s*부\s*칙\s*$")
_REVISION_TAIL = re.compile(r"\s*\((?:개정|신설|삭제)[^)]*\)\s*$")
_DATE_IN_NAME = re.compile(r"(\d{4})(\d{2})(\d{2})")


@dataclass
class Article:
    number: int
    branch: int | None  # 조의2 → 2
    title: str
    body: str
    chapter: str = ""  # 예: "제8장 휴·복학 및 퇴학"
    section: str = ""  # 예: "제3절 교과의 이수"

    @property
    def article_no(self) -> str:
        return f"제{self.number}조" + (f"의{self.branch}" if self.branch else "")


def base_date_from_name(path: Path) -> tuple[str, str]:
    """파일명 hakchik-20251001.txt → ('2025-10-01', '2025.10.1')."""
    m = _DATE_IN_NAME.search(path.name)
    if not m:
        raise ValueError(f"파일명에 시행일(YYYYMMDD)이 없습니다: {path.name}")
    y, mo, d = m.groups()
    return f"{y}-{mo}-{d}", f"{y}.{int(mo)}.{int(d)}"


def _squash(s: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _chapter_title(raw: str) -> str:
    """'총         칙' → '총칙', '학 생 활 동' → '학생활동', '(개정 …)' 꼬리 제거."""
    raw = _REVISION_TAIL.sub("", raw.strip())
    if re.search(r"\s{2,}", raw) and len(raw.split()) <= 3 and all(len(t) <= 2 for t in raw.split()):
        return "".join(raw.split())  # 글자 사이를 띄운 제목
    tokens = raw.split()
    if len(tokens) == 2 and len(tokens[0]) == 1 and "(" in tokens[1]:
        return "".join(tokens)  # '전 과(부)' → '전과(부)'
    if tokens and all(len(t) == 1 for t in tokens):
        return "".join(tokens)
    return _squash(raw)


def _clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in text.replace("\f", "\n").splitlines():
        if _NOISE.match(line):
            continue
        out.append(line.rstrip())
    return out


def parse_articles(text: str) -> list[Article]:
    lines = _clean_lines(text)
    articles: list[Article] = []
    chapter = section = ""
    cur: Article | None = None
    buf: list[str] = []
    started = False

    def flush() -> None:
        nonlocal cur, buf
        if cur is not None:
            cur.body = "\n".join(buf).strip()
            articles.append(cur)
        cur, buf = None, []

    for line in lines:
        stripped = line.strip()
        m = _ARTICLE.match(stripped)
        if m and not line.startswith(" "):  # 조 제목은 항상 줄 맨 앞에서 시작
            started = True
            flush()
            num, branch, title, rest = m.groups()
            cur = Article(
                number=int(num),
                branch=int(branch) if branch else None,
                title=_squash(title),
                body="",
                chapter=chapter,
                section=section,
            )
            buf = [_squash(rest)] if rest else []
            continue
        if not started:
            c0 = _CHAPTER.match(stripped)
            if c0 and c0.group(2) == "장":  # 제1조 앞의 '제 1 장 총칙'
                chapter = f"제{c0.group(1)}장 {_chapter_title(c0.group(3))}"
            continue
        if _BODY_END.match(stripped):
            break
        c = _CHAPTER.match(stripped)
        if c and line.startswith(" "):  # 장·절 제목은 가운데 정렬(앞에 공백)
            flush()
            label = f"제{c.group(1)}{c.group(2)} {_chapter_title(c.group(3))}"
            if c.group(2) == "장":
                chapter, section = label, ""
            else:
                section = label
            continue
        if cur is not None and stripped:  # 쪽 경계의 빈 줄은 버린다(조문 안에는 문단 사이 빈 줄이 없음)
            buf.append(_squash(line))
    flush()
    return [a for a in articles if not a.title.startswith("삭제")]


def _collapse_blank(body: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", body)


def article_title(a: Article) -> str:
    return f"학칙 {a.article_no}({a.title})"


def article_url(a: Article) -> str:
    return f"{RULES_URL}#학칙-{a.article_no}"


def to_pages(articles: list[Article], base_iso: str, base_label: str) -> list[rag.Page]:
    pages = []
    for a in articles:
        pages.append(
            rag.Page(
                slug=f"hakchik-{a.article_no}",
                title=article_title(a),
                source_url=article_url(a),
                body=_collapse_blank(a.body),
                doc_type=DOC_TYPE_REG,
                published_at=f"{base_iso}T00:00:00Z",
                category=" · ".join(p for p in (a.chapter, a.section) if p),
                date_label=base_label,
                article_no=a.article_no,
            )
        )
    return pages


def latest_file(directory: Path = REGULATIONS_DIR) -> Path:
    files = sorted(directory.glob("hakchik-*.txt"))
    if not files:
        raise FileNotFoundError(f"{directory}에 hakchik-YYYYMMDD.txt가 없습니다.")
    return files[-1]  # 파일명 날짜가 가장 늦은(최신) 학칙


def load_pages(path: Path | None = None) -> list[rag.Page]:
    path = path or latest_file()
    iso, label = base_date_from_name(path)
    articles = parse_articles(path.read_text(encoding="utf-8"))
    return to_pages(articles, iso, label)
