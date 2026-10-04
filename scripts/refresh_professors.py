"""학과 교수진 연락처 수집·갱신 도구 (1-4a).

사용법 (ai 폴더에서):
  python scripts/refresh_professors.py             # 학과 교수진 페이지를 읽어 이전 수집본과 비교하고, 달라졌으면 문서를 갱신
  python scripts/refresh_professors.py --check     # 비교만 하고 파일은 쓰지 않음 (달라졌으면 종료 코드 1)

학과 목록과 교수진 페이지 주소는 학교 조직도 데이터(`refresh_contacts.py`와 같은 출처)에서 가져온다.

결과:
  data/pages/manual/professors.md      교수 1명당 한 덩어리 (퇴임교수는 퇴임 사실만)
  data/pages/_professors_snapshot.json 비교용 수집본

문서를 갱신한 뒤에는 `python scripts/scrape_pages.py --slug professors`를 한 번 더 돌려
data/pages/professors.md(RAG 입력)에 반영한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import contacts as ct
from ai import professors as pf

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "data" / "pages"
SNAPSHOT = PAGES_DIR / "_professors_snapshot.json"
DOC = PAGES_DIR / "manual" / "professors.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="학과 교수진 연락처 갱신")
    parser.add_argument("--check", action="store_true", help="비교만 하고 파일은 쓰지 않음")
    args = parser.parse_args()

    try:
        sites = ct.build_sites(ct.fetch_sheet(ct.ORG_SHEET))
        new = pf.collect(sites)
    except (ct.ContactsError, pf.ProfessorsError) as exc:
        print(f"[실패] {exc}")
        return 2

    total = sum(len(p["professors"]) for p in new["pages"].values())
    retired = sum(1 for p in new["pages"].values() for x in p["professors"] if x["retired"])
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    changes = pf.diff_data(old, new) if old else ["첫 수집"]
    if old and not changes and DOC.exists():
        print(f"변경 없음 (학과 페이지 {len(new['pages'])}곳, 교수 {total}명)")
        return 0

    print(
        f"변경 {len(changes)}건 (학과 페이지 {len(new['pages'])}곳, 교수 {total}명, 퇴임 {retired}명):"
    )
    for line in changes[:60]:
        print(f"  - {line}")
    if len(changes) > 60:
        print(f"  ... 외 {len(changes) - 60}건")
    if args.check:
        return 1

    today = datetime.now(tz=timezone(timedelta(hours=9))).date().isoformat()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(pf.render_markdown(new, today), encoding="utf-8", newline="\n")
    SNAPSHOT.write_text(
        json.dumps(new, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"저장: {DOC.relative_to(ROOT)}")
    print("이어서: python scripts/scrape_pages.py --slug professors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
