"""홈페이지 안내 페이지 수집 도구 (1-4a).

사용법 (ai 폴더에서):
  python scripts/scrape_pages.py                 # tier 1(학사안내 메뉴·FAQ)을 가져와 data/pages/에 저장
  python scripts/scrape_pages.py --tier 1 --tier 2
  python scripts/scrape_pages.py --slug leave --slug graduation   # 일부만
  python scripts/scrape_pages.py --dry-run       # 가져와서 결과만 보고, 파일은 안 씀
  python scripts/scrape_pages.py --discover      # 사이트맵에 있는데 sources.json에 없는 주소 찾기
  python scripts/scrape_pages.py --url https://www.skuniv.ac.kr/xxx --slug xxx   # 목록에 없는 한 페이지

결과:
  data/pages/<slug>.md      본문(마크다운). 내용이 같으면 다시 쓰지 않음
  data/pages/_state.json    페이지별 수집 상태(시각 없음)
  data/pages/_report.md     한눈에 보는 보고서 + 사람이 확인·입력할 페이지 목록
  data/pages/manual/<slug>.md  사람이 복붙한 내용 — 있으면 사이트 추출보다 우선

사이트에는 1초에 1번만 요청한다(--delay). robots.txt와 User-Agent를 지킨다.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import page_scraper as ps

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "data" / "pages"


def discover(known_paths: set[str]) -> int:
    urls = ps.discover_sitemap()
    new = []
    for url in urls:
        path = "/" + ps.readable_slug(url) if ps.readable_slug(url) != "(home)" else "/"
        if path not in known_paths:
            new.append(url)
    print(f"사이트맵 {len(urls)}개 중 sources.json에 없는 주소 {len(new)}개:")
    for url in new:
        print("  ", ps.readable_slug(url), "  ", url)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="홈페이지 안내 페이지 수집 도구 (1-4a)")
    ap.add_argument("--tier", type=int, action="append", help="수집할 단계 (기본 1, 여러 번 가능)")
    ap.add_argument("--slug", action="append", help="이 slug만 (여러 번 가능)")
    ap.add_argument("--url", help="목록에 없는 한 페이지 (--slug 필요)")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 결과만 보고")
    ap.add_argument("--delay", type=float, default=1.0, help="요청 간격(초), 기본 1.0")
    ap.add_argument("--discover", action="store_true", help="사이트맵에서 새 주소 찾기")
    args = ap.parse_args(argv)

    sources = ps.load_sources(PAGES_DIR / "sources.json")
    if args.discover:
        return discover({str(s["path"]) for s in sources})

    if args.url:
        if not args.slug or len(args.slug) != 1:
            ap.error("--url은 --slug 하나와 함께 써야 합니다")
        targets: list[dict[str, object]] = [
            {"slug": args.slug[0], "path": args.url, "group": "직접 지정", "tier": 0}
        ]
    elif args.slug:
        wanted = set(args.slug)
        targets = [s for s in sources if s["slug"] in wanted]
        missing = wanted - {str(s["slug"]) for s in targets}
        if missing:
            ap.error(f"sources.json에 없는 slug: {', '.join(sorted(missing))}")
    else:
        tiers = set(args.tier or [1])
        targets = [s for s in sources if s["tier"] in tiers]

    robots = ps.load_robots()
    manual_dir = PAGES_DIR / "manual"
    records: list[ps.PageRecord] = []
    for i, src in enumerate(targets):
        slug, path = str(src["slug"]), str(src["path"])
        url = path if path.startswith("http") else ps.BASE_URL + path
        rec = ps.load_manual(slug, manual_dir, url)
        if rec is None:
            if i:
                time.sleep(args.delay)
            rec = ps.scrape_one(
                slug, url, robots=robots, group=str(src.get("group", "")), tier=int(str(src["tier"])),
                paginate=bool(src.get("paginate")), delay=args.delay,
            )
            hint = str(src.get("note", ""))
            if hint and rec.status != "ok":
                rec.note = f"{rec.note} ({hint})" if rec.note else hint
        rec.group, rec.tier = str(src.get("group", "")), int(str(src["tier"]))
        records.append(rec)
        print(f"[{rec.status:9}] {slug:34} {rec.chars:6}자  {rec.note}")

    ps.mark_duplicates(records)
    actions: dict[str, int] = {}
    for rec in records:
        if rec.status == "duplicate":
            print(f"[duplicate] {rec.slug}: {rec.note}")
        if args.dry_run or not rec.writable or rec.status == "duplicate":
            continue
        act = ps.write_record(rec, PAGES_DIR)
        actions[act] = actions.get(act, 0) + 1

    if args.dry_run:
        print("\n(dry-run: 파일을 쓰지 않았습니다)")
        return 0
    state = ps.update_state(PAGES_DIR / "_state.json", records, keep={str(s["slug"]) for s in sources})
    (PAGES_DIR / "_report.md").write_text(ps.render_report(state), encoding="utf-8")
    summary: dict[str, int] = {}
    for rec in records:
        summary[rec.status] = summary.get(rec.status, 0) + 1
    print("\n상태:", summary, "| 파일:", actions or "변경 없음")
    print("보고서: data/pages/_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
