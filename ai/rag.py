"""RAG 공통 파이프라인 — 안내 페이지 청킹·임베딩·적재·검색 (1-4a).

흐름: data/pages/<slug>.md(수집 결과) → 문서 읽기(load_pages) → 청킹(chunk_page)
      → 임베딩(embed_texts, Gemini 768차원) → DB 적재(sync_documents) → 검색(search).

- 문서 1개 = `admin_reg_documents` 1행(doc_type=`안내`), 청크 1개 = `admin_faq_embeddings` 1행.
- 청킹 규칙: 교수진은 교수 1명, FAQ는 질문 1개, 나머지는 제목(heading) 단위.
  너무 짧은 절은 다음 절과 합치고, 너무 긴 절은 문단 경계에서 자른다.
  모든 청크 맨 앞에 `[페이지 제목 > 소제목]`을 붙여 검색·답변 때 출처가 보이게 한다.
- 같은 내용이면 다시 임베딩하지 않는다(문서 본문 비교) — 여러 번 돌려도 안전.
- 임베딩·DB 라이브러리는 함수 안에서 import한다(없는 환경에서도 청킹 테스트가 돌도록).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EMBEDDING_DIM = 768
EMBEDDING_MODEL = "gemini-embedding-001"
DOC_TYPE_GUIDE = "안내"
DOC_TYPE_NOTICE = "공지"
DOC_TYPE_REGULATION = "학칙"
NOTICE_SHORT_CHARS = 80  # 공지 본문이 이보다 짧으면 "제목만 있는 공지"로 다룬다.

MIN_CHUNK_CHARS = 200  # 이보다 짧은 절은 다음 절과 합친다.
MAX_CHUNK_CHARS = 1500  # 이보다 긴 절은 문단 경계에서 나눈다.
EMBED_BATCH = 50

PAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "pages"
OFFICE_SLUG = "organization-phone"  # 조직도 페이지 — 같은 주소를 쓰는 다른 문서와 구분 기준

Embedder = Callable[[Sequence[str], str], list[list[float]]]


# ---------------------------------------------------------------- 문서 읽기


@dataclass
class Page:
    slug: str
    title: str
    source_url: str  # 최종 DB 값(중복이면 `#slug`가 붙은 상태는 assign_urls 이후)
    body: str
    status: str = "ok"
    doc_type: str = DOC_TYPE_GUIDE  # DB doc_type (안내/공지)
    published_at: str | None = None  # 공지 게시 시각(UTC ISO) — DB published_at
    category: str = ""  # 공지 분류(학사·장학 …)
    date_label: str = ""  # 공지 게시일(한국 날짜) 또는 학칙 시행일(2025.10.1), 청크 머리말용
    article_no: str | None = None  # 학칙 조 번호(제N조 / 제N조의2) — DB article_no


@dataclass
class Chunk:
    heading: str  # 예: "졸업·수료 > 조기졸업 > 자격"
    text: str  # 임베딩·답변에 쓰는 전체 글(맨 앞 [제목] 포함)
    ordinal: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    m = re.match(r"^---\n(.*?)\n---\n?", raw, flags=re.DOTALL)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"')
    return meta, raw[m.end() :].lstrip("\n")


def load_pages(pages_dir: Path = PAGES_DIR, *, include_short: bool = False) -> list[Page]:
    """data/pages/*.md 중 status가 ok인 문서만(include_short=True면 short도)."""
    allowed = {"ok", "short"} if include_short else {"ok"}
    pages: list[Page] = []
    for path in sorted(pages_dir.glob("*.md")):
        if path.name.startswith("_") or path.name == "README.md":
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if meta.get("status") not in allowed or not meta.get("source_url"):
            continue
        pages.append(
            Page(
                slug=meta.get("slug") or path.stem,
                title=meta.get("title") or path.stem,
                source_url=meta["source_url"],
                body=body.strip(),
                status=meta["status"],
            )
        )
    return assign_urls(pages)


def assign_urls(pages: list[Page]) -> list[Page]:
    """source_url은 DB에서 유일해야 한다. 같은 주소를 쓰는 문서가 여럿이면
    주소의 마지막 경로와 슬러그가 같은 문서(또는 가장 먼저 나온 문서)만 원래 주소를 갖고,
    나머지는 `#슬러그`를 붙인다(명세 5장)."""
    by_url: dict[str, list[Page]] = {}
    for p in pages:
        by_url.setdefault(p.source_url, []).append(p)
    for url, group in by_url.items():
        if len(group) == 1:
            continue
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        owner = next((p for p in group if p.slug == tail or p.slug == OFFICE_SLUG), group[0])
        for p in group:
            if p is not owner:
                p.source_url = f"{url}#{p.slug}"
    urls = [p.source_url for p in pages]
    if len(urls) != len(set(urls)):
        raise ValueError("source_url이 겹치는 문서가 있습니다.")
    return pages


# ---------------------------------------------------------------- 청킹


def _split_sections(body: str, level: int) -> list[tuple[list[str], str]]:
    """본문을 `#`×level 제목 단위로 나눈다. 반환: [(제목 경로, 본문), ...]."""
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    path: list[str] = []
    in_code = False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            depth = len(m.group(1))
            title = m.group(2).strip()
            path = path[: depth - 1]
            while len(path) < depth - 1:
                path.append("")
            path.append(title)
            if depth <= level:
                sections.append(([p for p in path if p], []))
                continue
        sections[-1][1].append(line)
    return [(p, "\n".join(lines).strip()) for p, lines in sections if "\n".join(lines).strip()]


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    cur = ""
    for para in re.split(r"\n{2,}", text):
        if cur and len(cur) + len(para) + 2 > limit:
            parts.append(cur)
            cur = ""
        while len(para) > limit:  # 문단 하나가 한도를 넘으면 줄 단위로
            piece = ""
            for line in para.split("\n"):
                if piece and len(piece) + len(line) + 1 > limit:
                    parts.append(piece)
                    piece = ""
                piece = f"{piece}\n{line}" if piece else line
            para = piece
            if len(para) <= limit:
                break
            parts.append(para[:limit])
            para = para[limit:]
        cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def _label(title: str, path: list[str]) -> str:
    trail = [title] + [p for p in path if p != title]
    return " > ".join(trail)


def _chunk_by_heading(page: Page) -> list[Chunk]:
    out: list[Chunk] = []
    pending_path: list[str] | None = None
    pending = ""
    for path, text in _split_sections(page.body, level=3):
        if pending and len(pending) < MIN_CHUNK_CHARS:
            text = f"{pending}\n\n{text}"
            path = pending_path or path  # 합쳐도 앞 절의 제목 경로를 유지
        elif pending:
            out.extend(_emit(page, pending_path or [], pending))
        pending, pending_path = text, path
    if pending:
        out.extend(_emit(page, pending_path or [], pending))
    return out


def _emit(page: Page, path: list[str], text: str) -> list[Chunk]:
    label = _label(page.title, path)
    return [Chunk(heading=label, text=f"[{label}]\n{piece}") for piece in _split_long(text)]


def _chunk_items(page: Page, item_prefix: str) -> list[Chunk]:
    """`### …` 한 항목 = 청크 1개(교수 1명, FAQ 1건). 항목 앞의 소개글은 별도 청크."""
    out: list[Chunk] = []
    group = ""  # 직전 `## 학과` 제목
    intro: list[str] = []
    cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_title, cur_lines
        body = "\n".join(cur_lines).strip()
        if cur_title and body:
            label = _label(page.title, [group, cur_title] if group else [cur_title])
            for piece in _split_long(body):
                out.append(Chunk(heading=label, text=f"[{label}]\n{piece}"))
        cur_title, cur_lines = "", []

    for line in page.body.splitlines():
        if line.startswith("## "):
            flush()
            group = line[3:].strip()
            continue
        if line.startswith(item_prefix):
            flush()
            cur_title = line[len(item_prefix) :].strip()
            continue
        if cur_title:
            cur_lines.append(line)
        else:
            intro.append(line)
    flush()
    head = "\n".join(intro).strip()
    if head and len(head) >= MIN_CHUNK_CHARS // 2:
        out.insert(0, Chunk(heading=page.title, text=f"[{page.title}]\n{head}"))
    return out


def _notice_head(page: Page) -> str:
    parts = ["공지", page.category, page.title, page.date_label]
    return "[" + " · ".join(p for p in parts if p) + "]"


def _readable_chars(text: str) -> int:
    from ai.page_scraper import count_chars

    return count_chars(text)


def _chunk_notice(page: Page) -> list[Chunk]:
    """공지 청킹. 모든 청크 맨 앞에 `[공지 · 분류 · 제목 · 날짜]`를 붙여 제목으로도 검색되게 한다.
    본문이 거의 없는 공지(포스터 이미지·첨부만 있는 글)는 제목만으로 한 청크를 만든다."""
    head = _notice_head(page)
    body = page.body.strip()
    if _readable_chars(body) < NOTICE_SHORT_CHARS:
        note = "본문 내용이 거의 없는 공지입니다(이미지·첨부 파일로 안내되는 경우가 많음). 자세한 내용은 원문 링크를 확인하세요."
        text = f"{head}\n{note}" + (f"\n{body}" if body else "")
        return [Chunk(heading=page.title, text=text, meta={"short": True})]
    return [Chunk(heading=page.title, text=f"{head}\n{piece}") for piece in _split_long(body)]


def _chunk_regulation(page: Page) -> list[Chunk]:
    """학칙 청킹: 조 1개 = 청크 1개(너무 길면 항 단위로 나눔). 맨 앞에 `[학칙 제N조(제목) · 장 · 절 · 기준일]`."""
    parts = [page.title, page.category, f"{page.date_label} 시행 기준" if page.date_label else ""]
    head = "[" + " · ".join(p for p in parts if p) + "]"
    return [
        Chunk(heading=page.title, text=f"{head}\n{piece}")
        for piece in _split_long(page.body.strip(), limit=MAX_CHUNK_CHARS)
    ]


def chunk_page(page: Page) -> list[Chunk]:
    """페이지 종류에 맞는 청킹. 학칙은 조별, 공지는 머리말+본문, 교수진·FAQ는 항목별, 나머지는 제목별."""
    if page.doc_type == DOC_TYPE_REGULATION:
        chunks = _chunk_regulation(page)
        for i, c in enumerate(chunks):
            c.ordinal = i
        return chunks
    if page.doc_type == DOC_TYPE_NOTICE:
        chunks = _chunk_notice(page)
        for i, c in enumerate(chunks):
            c.ordinal = i
        return chunks
    if page.slug == "professors":
        chunks = _chunk_items(page, "### ")
    elif page.slug == "all-faq":
        chunks = _chunk_items(page, "### Q. ")
    else:
        chunks = _chunk_by_heading(page)
    if not chunks and page.body:
        chunks = _emit(page, [], page.body)
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


# ---------------------------------------------------------------- 임베딩


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: Sequence[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Gemini 임베딩(768차원). task: 문서는 RETRIEVAL_DOCUMENT, 질문은 RETRIEVAL_QUERY."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    model = os.environ.get("GEMINI_EMBEDDING_MODEL") or EMBEDDING_MODEL
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch: list[Any] = list(texts[i : i + EMBED_BATCH])
        res = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM, task_type=task),
        )
        embeddings = res.embeddings or []
        if len(embeddings) != len(batch):
            raise RuntimeError("임베딩 응답 개수가 요청과 다릅니다.")
        for e in embeddings:
            values = list(e.values or [])
            if len(values) != EMBEDDING_DIM:
                raise RuntimeError(f"임베딩 차원이 {len(values)}입니다(기대 {EMBEDDING_DIM}).")
            out.append(_normalize(values))
    return out


def _vec_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


# ---------------------------------------------------------------- DB 적재·검색


@dataclass
class SyncResult:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0


def sync_documents(
    pages: list[Page],
    *,
    session: Any,
    embedder: Embedder = embed_texts,
    remove_missing: bool = False,
    doc_type: str = DOC_TYPE_GUIDE,
) -> SyncResult:
    """문서를 DB에 맞춘다(doc_type: 안내/공지). 제목·본문이 같으면 건너뛰고, 달라졌으면 청크를 다시 만든다.
    remove_missing=True면 이번 목록에 없는 같은 doc_type 문서를 지운다(청크는 CASCADE)."""
    from sqlalchemy import text

    result = SyncResult()
    existing = {
        row.source_url: row
        for row in session.execute(
            text(
                "SELECT id, source_url, title, content FROM admin_reg_documents "
                "WHERE doc_type = CAST(:t AS doc_type)"
            ),
            {"t": doc_type},
        )
    }
    for page in pages:
        chunks = chunk_page(page)
        if not chunks:
            continue
        row = existing.get(page.source_url)
        if row is not None and row.content == page.body and row.title == page.title:
            result.unchanged += 1
            continue
        vectors = embedder([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
        if row is None:
            doc_id = session.execute(
                text(
                    "INSERT INTO admin_reg_documents "
                    "(id, title, doc_type, article_no, content, source_url, published_at) "
                    "VALUES (gen_random_uuid(), :title, CAST(:t AS doc_type), :art, :content, :url, "
                    "CAST(:pub AS timestamptz)) RETURNING id"
                ),
                {
                    "title": page.title,
                    "t": doc_type,
                    "art": page.article_no,
                    "content": page.body,
                    "url": page.source_url,
                    "pub": page.published_at,
                },
            ).scalar_one()
            result.added += 1
        else:
            doc_id = row.id
            session.execute(
                text(
                    "UPDATE admin_reg_documents SET title = :title, content = :content, "
                    "article_no = :art, updated_at = now() WHERE id = :id"
                ),
                {"title": page.title, "content": page.body, "art": page.article_no, "id": doc_id},
            )
            session.execute(
                text("DELETE FROM admin_faq_embeddings WHERE document_id = :id"), {"id": doc_id}
            )
            result.updated += 1
        for chunk, vec in zip(chunks, vectors, strict=True):
            session.execute(
                text(
                    "INSERT INTO admin_faq_embeddings (id, document_id, embedding, chunk_text) "
                    "VALUES (gen_random_uuid(), :doc, CAST(:vec AS vector), :chunk)"
                ),
                {"doc": doc_id, "vec": _vec_literal(vec), "chunk": chunk.text},
            )
        result.chunks += len(chunks)
    if remove_missing:
        keep = {p.source_url for p in pages}
        for url, row in existing.items():
            if url not in keep:
                session.execute(text("DELETE FROM admin_reg_documents WHERE id = :id"), {"id": row.id})
                result.removed += 1
    return result


@dataclass
class Hit:
    chunk_text: str
    title: str
    doc_type: str
    article_no: str | None
    source_url: str | None
    distance: float  # 코사인 거리(작을수록 비슷함)
    published_at: str | None = None  # 공지 게시 시각(ISO)
    score: float = 0.0  # 순위 점수 = 거리 + 오래된 공지 가산점(작을수록 위)


RECENCY_PER_YEAR = 0.03  # 공지는 1년 오래될수록 거리에 이만큼 더한다(같은 내용이면 최신이 위로).
RECENCY_MAX_YEARS = 2.0  # 가산점은 최대 2년치까지만.
CANDIDATE_FACTOR = 4  # 순위를 다시 매기려고 k의 몇 배를 먼저 가져온다.


def _age_years(published_at: str, now: datetime) -> float:
    when = datetime.fromisoformat(published_at)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((now - when).total_seconds() / (365.25 * 86400), 0.0)


def rerank(hits: list[Hit], k: int, now: datetime | None = None) -> list[Hit]:
    """공지(게시일이 있는 문서)에만 오래된 만큼 가산점을 주고 점수 순으로 k개.
    학칙·안내 문서는 게시일이 없어 거리 그대로다."""
    current = now or datetime.now(UTC)
    for h in hits:
        penalty = 0.0
        if h.doc_type == DOC_TYPE_NOTICE and h.published_at:
            penalty = RECENCY_PER_YEAR * min(_age_years(h.published_at, current), RECENCY_MAX_YEARS)
        h.score = h.distance + penalty
    return sorted(hits, key=lambda h: h.score)[:k]


def search(
    query: str,
    *,
    session: Any,
    k: int = 5,
    doc_types: Sequence[str] | None = None,
    embedder: Embedder = embed_texts,
    now: datetime | None = None,
) -> list[Hit]:
    """질문과 가까운 청크 k개(공지는 최신일수록 유리). doc_types로 학칙/공지/안내를 골라 볼 수 있다."""
    from sqlalchemy import text

    vec = embedder([query], "RETRIEVAL_QUERY")[0]
    sql = (
        "SELECT e.chunk_text, d.title, d.doc_type::text AS doc_type, d.article_no, d.source_url, "
        "e.embedding <=> CAST(:vec AS vector) AS distance, d.published_at "
        "FROM admin_faq_embeddings e JOIN admin_reg_documents d ON d.id = e.document_id "
    )
    params: dict[str, Any] = {"vec": _vec_literal(vec), "k": k * CANDIDATE_FACTOR}
    if doc_types:
        sql += "WHERE d.doc_type::text = ANY(:types) "
        params["types"] = list(doc_types)
    sql += "ORDER BY e.embedding <=> CAST(:vec AS vector) LIMIT :k"
    hits = [
        Hit(
            r.chunk_text,
            r.title,
            r.doc_type,
            r.article_no,
            r.source_url,
            float(r.distance),
            r.published_at.isoformat() if r.published_at else None,
        )
        for r in session.execute(text(sql), params)
    ]
    return rerank(hits, k, now)


def chunk_report(pages: list[Page]) -> str:
    """청킹 결과 요약(검수용) — JSON 문자열."""
    rows = []
    for p in pages:
        sizes = [len(c.text) for c in chunk_page(p)]
        rows.append(
            {
                "slug": p.slug,
                "url": p.source_url,
                "chunks": len(sizes),
                "min": min(sizes, default=0),
                "max": max(sizes, default=0),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=1)
