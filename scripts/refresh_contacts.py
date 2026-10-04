"""조직도·부서 연락처 갱신 도구 (1-4a).

사용법 (ai 폴더에서):
  python scripts/refresh_contacts.py             # 사이트에서 받아 이전 수집본과 비교하고, 달라졌으면 문서를 갱신
  python scripts/refresh_contacts.py --check     # 비교만 하고 파일은 쓰지 않음 (달라졌으면 종료 코드 1)

결과:
  data/pages/manual/organization-phone.md   부서별 연락처 문서 (달라졌을 때만 다시 씀)
  data/pages/manual/department-sites.md     학과 홈페이지·교수진 페이지 주소 + 대학원 안내
  data/pages/_contacts_snapshot.json        비교용 수집본(개인 성명·이메일 없음)

문서를 갱신한 뒤에는 `python scripts/scrape_pages.py --slug organization-phone --slug department-sites`를 한 번 더 돌려
data/pages/*.md(RAG 입력)에 반영한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import contacts as ct

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "data" / "pages"
SNAPSHOT = PAGES_DIR / "_contacts_snapshot.json"
MANUAL_DOC = PAGES_DIR / "manual" / "organization-phone.md"
SITES_DOC = PAGES_DIR / "manual" / "department-sites.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="조직도·부서 연락처 갱신")
    parser.add_argument("--check", action="store_true", help="비교만 하고 파일은 쓰지 않음")
    args = parser.parse_args()

    try:
        contacts = ct.fetch_sheet(ct.CONTACTS_SHEET)
        org = ct.fetch_sheet(ct.ORG_SHEET)
        new = ct.build_snapshot(contacts, org)
        ct.check_snapshot(new)
    except ct.ContactsError as exc:
        print(f"[실패] {exc}")
        return 2

    old = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
    changes = ct.diff_snapshots(old, new) if old else ["첫 수집"]
    if old and not changes and MANUAL_DOC.exists() and SITES_DOC.exists():
        print(f"변경 없음 (부서 {len(new['departments'])}곳)")
        return 0

    print(f"변경 {len(changes)}건:")
    for line in changes:
        print(f"  - {line}")
    if args.check:
        return 1

    MANUAL_DOC.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone(timedelta(hours=9))).date().isoformat()
    MANUAL_DOC.write_text(ct.render_markdown(new, today), encoding="utf-8", newline="\n")
    SITES_DOC.write_text(ct.render_sites_markdown(new, today), encoding="utf-8", newline="\n")
    SNAPSHOT.write_text(
        json.dumps(new, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"저장: {MANUAL_DOC.relative_to(ROOT)}, {SITES_DOC.relative_to(ROOT)}")
    print(
        "이어서: python scripts/scrape_pages.py --slug organization-phone --slug department-sites"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
