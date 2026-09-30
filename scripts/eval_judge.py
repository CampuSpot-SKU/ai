"""신고 판정 정확도 평가 — 카테고리·영향도·긴급도·우선순위를 실제 Gemini와 규칙 기반 비교군으로 비교 (명세서 10-1, 10-4 S-7).

실행: GitHub Actions → 'Judge eval' → Run workflow (수동, GEMINI_API_KEY 시크릿 사용)
  - set=holdout (기본): 평가 세트 — 결과보고서·영상에 쓰는 수치. 이 결과를 보고 프롬프트를 고치지 않음
  - set=dev: 개발 세트 — 프롬프트를 고친 뒤 회귀 확인용
  - --rules-only: Gemini 없이 규칙 기반만
채점: 카테고리·우선순위는 "문제언급=true"인 문장만, 영향도·긴급도는 전체. 우선순위는 명세 3-1 기본 매트릭스로
영향도×긴급도에서 계산 (P1 고×고 / P2 고×저 / P3 저×고 / P4 저×저).
평가 세트가 프롬프트 예시·개발 세트와 겹치면 Gemini를 부르기 전에 실패로 끝남.
"""
import argparse
import csv
import datetime as dt
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baseline_judge_rules import judge_rules
from intent_eval_lib import find_overlaps, pct

ROOT = Path(__file__).resolve().parent.parent
SETS = {
    "dev": ROOT / "tests" / "golden" / "judge_dev.csv",
    "holdout": ROOT / "tests" / "golden" / "judge_eval_holdout.csv",
}
PROMPT_PATH = ROOT / "ai" / "prompts" / "report_judge.md"
CATEGORIES = ["전기", "시설·설비", "청소·위생", "안전", "IT·네트워크", "기타"]
COLUMNS = ["발화", "위치", "기대카테고리", "기대영향도", "기대긴급도", "문제언급", "비고"]
MATRIX = {("high", "high"): "P1", ("high", "low"): "P2", ("low", "high"): "P3", ("low", "low"): "P4"}


@dataclass(frozen=True)
class Case:
    text: str
    location: str | None
    category: str
    impact: str
    urgency: str
    problem: bool
    note: str

    @property
    def priority(self) -> str:
        return MATRIX[(self.impact, self.urgency)]


def load_cases(path: Path) -> list[Case]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path.name}: 열 이름이 {COLUMNS}이어야 함 (지금 {reader.fieldnames})")
        cases = [
            Case(r["발화"], r["위치"] or None, r["기대카테고리"], r["기대영향도"], r["기대긴급도"],
                 r["문제언급"] == "true", r["비고"])
            for r in reader
        ]
    for c in cases:
        if c.category not in CATEGORIES or c.impact not in ("high", "low") or c.urgency not in ("high", "low"):
            raise ValueError(f"{path.name}: 값이 올바르지 않음: {c.text!r}")
    return cases


def prompt_examples(prompt_text: str) -> list[str]:
    """프롬프트 '## 예시' 이후 목록 줄의 첫 따옴표 문장(학생 말)."""
    _, _, examples = prompt_text.partition("## 예시")
    found = []
    for line in examples.splitlines():
        if line.lstrip().startswith("-"):
            m = re.search(r'"([^"]+)"', line)
            if m:
                found.append(m.group(1))
    return found


def check_overlap(cases: list[Case]) -> list[tuple[str, str, float]]:
    refs = [c.text for c in load_cases(SETS["dev"])]
    refs += prompt_examples(PROMPT_PATH.read_text(encoding="utf-8"))
    return find_overlaps([c.text for c in cases], refs)


def rules_predictor(case: Case) -> dict[str, Any]:
    return judge_rules(case.text, case.location)


def gemini_predictor(case: Case) -> dict[str, Any]:
    from ai.judge import JudgeRequest, judge  # Gemini 의존성은 여기서만

    try:
        res = judge(JudgeRequest(text=case.text, location=case.location, categories=CATEGORIES))
    except Exception as e:  # noqa: BLE001 — 실패도 결과표에 남김
        return {"error": repr(e)[:80]}
    return res.model_dump()


def _ok(pred: dict[str, Any], case: Case, field: str) -> bool:
    if field == "priority":
        return "error" not in pred and MATRIX[(pred["impact"], pred["urgency"])] == case.priority
    key = {"category": "category", "impact": "impact", "urgency": "urgency"}[field]
    return pred.get(key) == getattr(case, key)


def build_report(set_name: str, cases: list[Case], preds: dict[str, list[dict[str, Any]]]) -> str:
    model = os.environ.get("GEMINI_JUDGE_MODEL") or "gemini-3.5-flash-lite (기본값)"
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    graded = [i for i, c in enumerate(cases) if c.problem]
    out = [
        f"# 신고 판정 평가 — {set_name} 세트 ({len(cases)}문장)",
        "",
        f"- 실행: {kst} · 커밋 `{sha}` · 모델 {model}",
        "- 구성: " + " / ".join(f"{k} {v}" for k, v in Counter(c.category for c in cases).items())
        + f" · 문제 내용 없음(모호) {sum(not c.problem for c in cases)}문장",
        "- 채점: 카테고리·우선순위는 문제 내용이 있는 문장만, 영향도·긴급도는 전체",
        "",
        "## 항목별 정확도",
        "| 항목 | " + " | ".join(preds) + " |",
        "|---|" + "---|" * len(preds),
    ]
    rows = [
        ("카테고리", "category", graded),
        ("영향도", "impact", list(range(len(cases)))),
        ("긴급도", "urgency", list(range(len(cases)))),
        ("**우선순위 (P1~P4)**", "priority", graded),
    ]
    for label, field, idx in rows:
        cells = []
        for p in preds.values():
            ok = sum(_ok(p[i], cases[i], field) for i in idx)
            cells.append(f"{pct(ok / len(idx))} ({ok}/{len(idx)})")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for p in preds.values():
        ok = sum(("problem_stated" in p[i] and p[i]["problem_stated"] == cases[i].problem) for i in range(len(cases)))
        cells.append(f"{pct(ok / len(cases))} ({ok}/{len(cases)})")
    out.append("| 문제 내용 말했는지 | " + " | ".join(cells) + " |")

    out += ["", "## 긴급도 high(안전) 탐지 — 놓치면 안 되는 쪽", "| 지표 | " + " | ".join(preds) + " |",
            "|---|" + "---|" * len(preds)]
    for name_, fn in (("재현율 (진짜 긴급을 찾아냄)", "recall"), ("정밀도 (긴급이라 한 것 중 진짜)", "precision")):
        cells = []
        for p in preds.values():
            tp = sum(p[i].get("urgency") == "high" and cases[i].urgency == "high" for i in range(len(cases)))
            base = sum(cases[i].urgency == "high" for i in range(len(cases))) if fn == "recall" else sum(
                p[i].get("urgency") == "high" for i in range(len(cases)))
            cells.append(f"{pct(tp / base if base else 0)} ({tp}/{base})")
        out.append(f"| {name_} | " + " | ".join(cells) + " |")

    names = list(preds)
    if len(names) == 2:
        rules, ai = names
        wins = [i for i in graded if not _ok(preds[rules][i], cases[i], "priority") and _ok(preds[ai][i], cases[i], "priority")]
        losses = [i for i in graded if _ok(preds[rules][i], cases[i], "priority") and not _ok(preds[ai][i], cases[i], "priority")]
        out += ["", f"## 우선순위 {rules} ❌ → {ai} ✅ ({len(wins)}건)"] + _rows(cases, preds, wins)
        out += ["", f"## 우선순위 {rules} ✅ → {ai} ❌ ({len(losses)}건)"] + _rows(cases, preds, losses)
        wrong = [i for i in graded if not _ok(preds[ai][i], cases[i], "priority") and i not in losses]
        out += ["", f"## 둘 다 틀린 우선순위 ({len(wrong)}건)"] + _rows(cases, preds, wrong)
    out += ["", "## 전체 결과 (AI 판정 이유 포함)"] + _rows(cases, preds, range(len(cases)), with_reason=True)
    return "\n".join(out) + "\n"


def _cell(p: dict[str, Any]) -> str:
    if "error" in p:
        return f"ERROR {p['error']}"
    return f"{p['category']} / {p['impact']} / {p['urgency']} → {MATRIX[(p['impact'], p['urgency'])]}"


def _rows(cases: list[Case], preds: dict[str, list[dict[str, Any]]], idx: Any, with_reason: bool = False) -> list[str]:
    names = list(preds)
    head = "| 발화 | 위치 | 기대 | " + " | ".join(names) + (" | AI 이유 |" if with_reason else " |")
    lines = [head, "|---|---|---|" + "---|" * len(names) + ("---|" if with_reason else "")]
    for i in idx:
        c = cases[i]
        exp = f"{c.category} / {c.impact} / {c.urgency} → {c.priority}" if c.problem else "(문제 내용 없음)"
        cells = []
        for p in preds.values():
            mark = "✅" if (not c.problem and p[i].get("problem_stated") is False) or (
                c.problem and _ok(p[i], c, "priority") and _ok(p[i], c, "category")) else "❌"
            cells.append(f"{mark} {_cell(p[i])}")
        reason = f" {preds[names[-1]][i].get('reason', '')} |" if with_reason else ""
        lines.append(f"| {c.text.replace(chr(10), ' ⏎ ')} | {c.location or '-'} | {exp} | " + " | ".join(cells) + " |" + reason)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=sorted(SETS), default="holdout")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--out", default="judge-eval-result.md")
    args = parser.parse_args()

    cases = load_cases(SETS[args.set])
    if args.set == "holdout" and (hits := check_overlap(cases)):
        print("평가 세트에 프롬프트 예시·개발 세트와 겹치는 문장이 있음 — 다른 문장으로 바꿀 것:")
        for t, r, s in hits:
            print(f"  {s:.2f}  {t}  ≈  {r}")
        return 1

    preds: dict[str, list[dict[str, Any]]] = {"규칙 기반": [rules_predictor(c) for c in cases]}
    if not args.rules_only:
        preds["AI (Gemini)"] = [gemini_predictor(c) for c in cases]
    report = build_report(args.set, cases, preds)
    print(report)
    Path(args.out).write_text(report, encoding="utf-8")
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(path).write_text(report, encoding="utf-8")
    errors = sum("error" in p for ps in preds.values() for p in ps)
    if errors:
        print(f"Gemini 호출 실패 {errors}건 — API 키·크레딧을 확인하고 다시 실행 (이 결과는 쓰지 말 것)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
