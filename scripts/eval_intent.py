"""의도 분류 정확도 평가 — tests/golden/intent_classification.csv를 실제 Gemini로 돌려봄.

실행: GitHub Actions → 'Intent eval' 워크플로우 → Run workflow (수동, GEMINI_API_KEY 시크릿 사용)
프롬프트를 고칠 때마다 다시 돌려서 정확도가 떨어지지 않았는지(회귀) 확인하는 용도 (명세서 10-1).
결과표는 Actions 실행 화면의 Summary에 표시됨.
"""
import csv
import os
import sys
from pathlib import Path

from ai.intent import HistoryMessage, classify

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "intent_classification.csv"


def parse_history(raw: str) -> list[HistoryMessage]:
    msgs = []
    for part in filter(None, (p.strip() for p in raw.split(" / "))):
        speaker, _, content = part.partition(":")
        role = "assistant" if speaker.strip() == "챗봇" else "user"
        msgs.append(HistoryMessage(role=role, content=content.strip()))
    return msgs


def main() -> int:
    rows = list(csv.DictReader(GOLDEN.open(encoding="utf-8")))
    out = ["| 결과 | 발화 | 기대 | 판정 | report/inquiry |", "|---|---|---|---|---|"]
    correct = 0
    for r in rows:
        try:
            res = classify(r["발화"], parse_history(r["이전대화"]))
            got, scores = res.intent, f"{res.report_score}/{res.inquiry_score}"
        except Exception as e:  # noqa: BLE001
            got, scores = "ERROR", repr(e)[:40]
        ok = got == r["기대판정"]
        correct += ok
        out.append(f"| {'✅' if ok else '❌'} | {r['발화']} | {r['기대판정']} | {got} | {scores} |")
    summary = f"## 의도 분류 정확도: {correct}/{len(rows)} ({correct / len(rows):.0%})\n\n"
    report = summary + "\n".join(out) + "\n"
    print(report)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(path).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
