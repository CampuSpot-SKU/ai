"""의도 분류 정확도 평가 — 정답 세트를 실제 Gemini와 규칙 기반 비교군으로 각각 돌려 비교 (명세서 10-1, 10-4 S-7a).

실행: GitHub Actions → 'Intent eval' → Run workflow (수동, GEMINI_API_KEY 시크릿 사용)
  - set=holdout (기본): 평가 세트 — 결과보고서·영상에 쓰는 수치. 이 결과를 보고 프롬프트를 고치지 않음
  - set=dev: 개발 세트 — 프롬프트를 고친 뒤 회귀 확인용 (프롬프트 예시와 겹치는 문장이 있어 수치가 높게 나옴)
  - --rules-only: Gemini 없이 규칙 기반만 (키 없이도 실행 가능)
결과는 Actions 실행 화면 Summary + 아티팩트(intent-eval-result.md)로 남음.
평가 세트가 프롬프트 예시·개발 세트와 겹치면 Gemini를 부르기 전에 실패로 끝남.
"""

import argparse
import datetime as dt
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from baseline_intent_rules import classify_rules
from intent_eval_lib import (
    LABEL_KO,
    PROMPT_PATH,
    SETS,
    Case,
    Metrics,
    compute_metrics,
    confusion_table,
    find_overlaps,
    load_cases,
    metrics_table,
    pct,
    prompt_examples,
)

Predictor = Callable[[Case], tuple[str, str]]  # (판정, 점수 등 메모)


def rules_predictor(case: Case) -> tuple[str, str]:
    return classify_rules(case.text, case.history), ""


def gemini_predictor(case: Case) -> tuple[str, str]:
    from ai.intent import HistoryMessage, classify  # Gemini 의존성은 여기서만

    try:
        history = [
            HistoryMessage.model_validate({"role": r, "content": c}) for r, c in case.history
        ]
        res = classify(case.text, history)
    except Exception as e:  # noqa: BLE001 — 실패도 결과표에 오답(ERROR)으로 남김
        return "ERROR", repr(e)[:60]
    safety = " ⚠️" if res.safety_concern else ""
    # 인사·잡담·범위 밖은 "신고도 문의도 아님"이라 평가 세트의 unclear(되묻기)와 같은 쪽으로 친다
    label = "unclear" if res.intent in ("chitchat", "off_topic") else res.intent
    return label, f"{res.report_score}/{res.inquiry_score}{safety}"


def check_overlap(cases: list[Case]) -> list[tuple[str, str, float]]:
    refs = [c.text for c in load_cases(SETS["dev"])]
    refs += prompt_examples(PROMPT_PATH.read_text(encoding="utf-8"))
    return find_overlaps([c.text for c in cases], refs)


def subgroup_line(
    name: str, cases: list[Case], preds: dict[str, list[str]], keep: Callable[[Case], bool]
) -> str:
    idx = [i for i, c in enumerate(cases) if keep(c)]
    if not idx:
        return ""
    cells = []
    for p in preds.values():
        ok = sum(p[i] == cases[i].expected for i in idx)
        cells.append(f"{pct(ok / len(idx))} ({ok}/{len(idx)})")
    return f"| {name} | " + " | ".join(cells) + " |"


def build_report(
    set_name: str, cases: list[Case], results: dict[str, tuple[list[str], list[str]]]
) -> str:
    preds = {name: p for name, (p, _) in results.items()}
    metrics: dict[str, Metrics] = {
        name: compute_metrics([c.expected for c in cases], p) for name, p in preds.items()
    }
    model = os.environ.get("GEMINI_INTENT_MODEL") or "gemini-3.5-flash-lite (기본값)"
    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    kst = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    counts = {label: sum(c.expected == label for c in cases) for label in LABEL_KO}
    out = [
        f"# 의도 분류 평가 — {set_name} 세트 ({len(cases)}문장)",
        "",
        f"- 실행: {kst} · 커밋 `{sha}` · 모델 {model}",
        "- 구성: "
        + " / ".join(f"{LABEL_KO[k]} {v}" for k, v in counts.items())
        + f" · 이전 대화 포함 {sum(bool(c.history_raw) for c in cases)}문장",
        "",
        "## 전체 지표",
        metrics_table(metrics),
        "",
        "## 문장 유형별 정확도",
        "| 유형 | " + " | ".join(preds) + " |",
        "|---|" + "---|" * len(preds),
    ]
    groups = [
        ("이전 대화 없음", lambda c: not c.history_raw),
        ("이전 대화 있음 (맥락 판단)", lambda c: bool(c.history_raw)),
        ("키워드 함정 (신고/문의 단어가 반대로 섞임)", lambda c: "키워드 함정" in c.note),
        ("신고+문의 섞임", lambda c: "+" in c.note),
        ("안전 관련", lambda c: "안전" in c.note),
    ]
    out += [line for n, k in groups if (line := subgroup_line(n, cases, preds, k))]
    for name, m in metrics.items():
        out += ["", f"## 혼동행렬 — {name}", confusion_table(m)]

    names = list(results)
    if len(names) == 2:
        rules, ai = names
        hit = {n: [p == c.expected for p, c in zip(preds[n], cases, strict=True)] for n in names}
        wins = [i for i in range(len(cases)) if not hit[rules][i] and hit[ai][i]]
        losses = [i for i in range(len(cases)) if hit[rules][i] and not hit[ai][i]]
        out += ["", f"## {rules} ❌ → {ai} ✅ ({len(wins)}건)"]
        out += _rows(cases, results, wins)
        out += ["", f"## {rules} ✅ → {ai} ❌ ({len(losses)}건)"]
        out += _rows(cases, results, losses)
    out += ["", "## 전체 결과"]
    out += _rows(cases, results, range(len(cases)))
    return "\n".join(out) + "\n"


def _rows(
    cases: list[Case], results: dict[str, tuple[list[str], list[str]]], idx: Iterable[int]
) -> list[str]:
    names = list(results)
    head = "| 발화 | 이전 대화 | 기대 | " + " | ".join(names) + " | 비고 |"
    lines = [head, "|---|---|---|" + "---|" * len(names) + "---|"]
    for i in idx:
        c = cases[i]
        cells = []
        for p, memo in results.values():
            mark = "✅" if p[i] == c.expected else "❌"
            cells.append(f"{mark} {LABEL_KO.get(p[i], p[i])}" + (f" {memo[i]}" if memo[i] else ""))
        lines.append(
            f"| {c.text} | {c.history_raw or '-'} | {LABEL_KO[c.expected]} | "
            + " | ".join(cells)
            + f" | {c.note} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=sorted(SETS), default="holdout")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--out", default="intent-eval-result.md")
    args = parser.parse_args()

    cases = load_cases(SETS[args.set])
    if args.set == "holdout" and (hits := check_overlap(cases)):
        print("평가 세트에 프롬프트 예시·개발 세트와 겹치는 문장이 있음 — 다른 문장으로 바꿀 것:")
        for t, r, s in hits:
            print(f"  {s:.2f}  {t}  ≈  {r}")
        return 1

    predictors: dict[str, Predictor] = {"규칙 기반": rules_predictor}
    if not args.rules_only:
        predictors["AI (Gemini)"] = gemini_predictor
    results: dict[str, tuple[list[str], list[str]]] = {}
    for name, fn in predictors.items():
        outs = [fn(c) for c in cases]
        results[name] = ([o[0] for o in outs], [o[1] for o in outs])

    report = build_report(args.set, cases, results)
    print(report)
    Path(args.out).write_text(report, encoding="utf-8")
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(path).write_text(report, encoding="utf-8")
    errors = sum(p == "ERROR" for preds, _ in results.values() for p in preds)
    if errors:
        print(
            f"Gemini 호출 실패 {errors}건 — API 키·크레딧을 확인하고 다시 실행 (이 결과는 쓰지 말 것)"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
