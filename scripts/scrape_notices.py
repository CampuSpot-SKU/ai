"""학교 공지 수집 도구 (1-14).

사용법 (ai 폴더에서):
  python scripts/scrape_notices.py                      # 최근 1년 공지를 받아 data/notices/notices.jsonl에 저장
  python scripts/scrape_notices.py --since 2025-10-06   # 날짜 지정(그 날 이후 전부)
  python scripts/scrape_notices.py --dry-run            # 받아서 요약만 보고, 파일은 안 씀
  python scripts/scrape_notices.py --incremental        # 저장된 마지막 게시일 이후 새 공지만 받아 합침

결과: data/notices/notices.jsonl (공지 1건 = 한 줄 JSON — 제목·분류·게시일·원문 주소·본문)
이어서 DB에 넣으려면 Actions "Embed pages"에서 target=notices, action=apply.
사이트에는 1초에 1번만 요청한다. 대외활동·채용공지·FYP를 포함한 8개 분류를 모두 받는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import notice_crawler as nc


def main() -> int:
    parser = argparse.ArgumentParser(description="학교 공지 수집")
    parser.add_argument("--since", help="이 날짜(YYYY-MM-DD) 이후 공지 (기본: 1년 전)")
    parser.add_argument("--dry-run", action="store_true", help="요약만 보고 파일은 안 씀")
    parser.add_argument("--incremental", action="store_true", help="저장된 마지막 게시일 이후만")
    args = parser.parse_args()

    old = nc.load_jsonl()
    since = args.since or nc.default_since()
    if args.incremental:
        if not old:
            print("[실패] 저장된 공지가 없어 증분 수집을 할 수 없습니다. 먼저 전체 수집을 하세요.")
            return 2
        since = old[-1]["date"][:10]  # 마지막 게시일 당일부터(겹치는 것은 합칠 때 덮어씀)
    try:
        new = nc.collect(since)
    except nc.NoticeError as exc:
        print(f"[실패] {exc}")
        return 2

    merged = nc.merge(old, new) if (args.incremental or old) and not args.since else nc.merge([], new)
    print(f"받은 공지 {len(new)}건 (기준일 {since} 이후)")
    for line in nc.summarize(merged):
        print(line)
    if args.dry_run:
        return 0
    nc.save_jsonl(merged)
    print(f"저장: data/notices/notices.jsonl ({len(merged)}건)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
