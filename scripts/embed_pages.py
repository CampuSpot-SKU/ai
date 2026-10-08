"""홈페이지 안내 페이지 임베딩·적재 도구 (1-4a).

사용법 (ai 폴더에서):
  python scripts/embed_pages.py                      # 청킹 결과만 확인 (DB·API 안 씀)
  python scripts/embed_pages.py --apply              # 임베딩해서 DB에 적재 (DATABASE_URL·GEMINI_API_KEY 필요)
  python scripts/embed_pages.py --apply --remove-missing   # 이번 목록에 없는 '안내' 문서는 DB에서 삭제
  python scripts/embed_pages.py --search "휴학 신청 기간"    # 검색 확인 (DATABASE_URL·GEMINI_API_KEY 필요)
  python scripts/embed_pages.py --notices [--apply]  # 안내 페이지 대신 공지(data/notices/notices.jsonl)를 처리
  python scripts/embed_pages.py --regulations [--apply]  # 학칙(data/regulations/hakchik-*.txt, 조 단위)을 처리

입력은 data/pages/*.md 중 status가 ok인 문서. 본문이 달라지지 않은 문서는 다시 임베딩하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import rag


def main() -> int:
    parser = argparse.ArgumentParser(description="안내 페이지 임베딩·적재")
    parser.add_argument("--apply", action="store_true", help="임베딩해서 DB에 적재")
    parser.add_argument("--remove-missing", action="store_true", help="목록에 없는 안내 문서 삭제")
    parser.add_argument("--search", metavar="질문", help="적재된 안내 문서에서 검색")
    parser.add_argument("--include-short", action="store_true", help="status=short 문서도 포함")
    parser.add_argument("--notices", action="store_true", help="공지(notices.jsonl)를 처리")
    parser.add_argument("--regulations", action="store_true", help="학칙(조 단위)을 처리")
    args = parser.parse_args()

    if args.search:
        from ai.db import get_session

        with get_session() as db:
            kind = (
                rag.DOC_TYPE_NOTICE
                if args.notices
                else rag.DOC_TYPE_REGULATION
                if args.regulations
                else rag.DOC_TYPE_GUIDE
            )
            for hit in rag.search(args.search, session=db, doc_types=[kind]):
                when = f" ({hit.published_at[:10]})" if hit.published_at and not args.regulations else ""
                print(f"[{hit.distance:.3f}] {hit.title}{when} — {hit.source_url}")
                print("   " + hit.chunk_text.replace("\n", " ")[:160])
        return 0

    doc_type = rag.DOC_TYPE_GUIDE
    if args.notices:
        from ai import notice_crawler as nc

        pages = nc.to_pages(nc.load_jsonl())
        doc_type = rag.DOC_TYPE_NOTICE
    elif args.regulations:
        from ai import regulations

        pages = regulations.load_pages()
        doc_type = rag.DOC_TYPE_REGULATION
    else:
        pages = rag.load_pages(include_short=args.include_short)
    chunks = sum(len(rag.chunk_page(p)) for p in pages)
    print(f"문서 {len(pages)}개 → 청크 {chunks}개")
    if not args.apply:
        if args.regulations:
            for p in pages[:3]:
                print(f"  예시 {p.title} → {rag.chunk_page(p)[0].text.splitlines()[0]}")
            pages = []
        if args.notices:
            short = sum(1 for p in pages for c in rag.chunk_page(p) if c.meta.get("short"))
            print(f"  이 중 제목만으로 검색되는 공지(본문 거의 없음): {short}건")
            pages = []
        for p in pages:
            sizes = [len(c.text) for c in rag.chunk_page(p)]
            print(f"  {p.slug:32s} 청크 {len(sizes):3d}  (글자 {min(sizes, default=0)}~{max(sizes, default=0)})")
        print("DB에 넣으려면 --apply")
        return 0

    from ai.db import get_session

    with get_session() as db:
        res = rag.sync_documents(
            pages, session=db, remove_missing=args.remove_missing, doc_type=doc_type
        )
    print(
        f"적재 완료: 추가 {res.added}, 갱신 {res.updated}, 변경 없음 {res.unchanged}, "
        f"삭제 {res.removed} (새로 만든 청크 {res.chunks}개)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
