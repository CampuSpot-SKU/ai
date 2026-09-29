"""의도 분류 평가 공통 로직 (S-7a, 명세서 10-4) — 표준 라이브러리만 사용해 CI 유닛테스트에서도 돌아감.

- 정답 세트 읽기: 개발 세트(intent_classification.csv, 프롬프트 수정·회귀 확인용)
  / 평가 세트(intent_eval_holdout.csv, 보고서 수치용 — 이걸 보고 프롬프트를 고치지 않음)
- 겹침 검사: 평가 세트 문장이 프롬프트 예시·개발 세트와 거의 같으면 "처음 보는 문장" 평가가 아니게 되므로 실패
- 지표: 정확도, 유형별 정밀도·재현율·F1, 혼동행렬
"""

import csv
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
SETS = {
    "dev": GOLDEN_DIR / "intent_classification.csv",
    "holdout": GOLDEN_DIR / "intent_eval_holdout.csv",
}
PROMPT_PATH = ROOT / "ai" / "prompts" / "intent_classification.md"
LABELS = ("report", "inquiry", "unclear")
LABEL_KO = {"report": "신고", "inquiry": "문의", "unclear": "애매"}
COLUMNS = ["발화", "이전대화", "기대판정", "비고"]
# 공백·문장부호를 뺀 두 문장의 유사도가 이 값 이상이면 "겹침"으로 봄
OVERLAP_THRESHOLD = 0.7


@dataclass(frozen=True)
class Case:
    text: str
    history_raw: str
    expected: str
    note: str

    @property
    def history(self) -> list[tuple[str, str]]:
        """'챗봇: ... / 학생: ...' → [(role, content)] (role: user | assistant)."""
        out = []
        for part in filter(None, (p.strip() for p in self.history_raw.split(" / "))):
            speaker, _, content = part.partition(":")
            out.append(("assistant" if speaker.strip() == "챗봇" else "user", content.strip()))
        return out


def load_cases(path: Path) -> list[Case]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"{path.name}: 열 이름이 {COLUMNS}이어야 함 (지금 {reader.fieldnames})"
            )
        cases = [Case(r["발화"], r["이전대화"], r["기대판정"], r["비고"]) for r in reader]
    bad = [c.text for c in cases if c.expected not in LABELS]
    if bad:
        raise ValueError(f"{path.name}: 기대판정이 {LABELS} 중 하나가 아님: {bad}")
    return cases


def prompt_examples(prompt_text: str) -> list[str]:
    """프롬프트 '## 예시' 이후 목록 줄에 따옴표로 들어간 문장(발화·이전 대화) 전부."""
    _, _, examples = prompt_text.partition("## 예시")
    found: list[str] = []
    for line in examples.splitlines():
        if line.lstrip().startswith("-"):
            found += re.findall(r'"([^"]+)"', line)
    return found


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def find_overlaps(
    targets: Sequence[str], references: Iterable[str], threshold: float = OVERLAP_THRESHOLD
) -> list[tuple[str, str, float]]:
    """targets 중 references(또는 targets 안의 다른 문장)와 너무 비슷한 쌍. 짧은 문장(3자 이하)은 제외."""
    refs = list(references)
    hits = []
    for i, t in enumerate(targets):
        if len(_norm(t)) <= 3:
            continue
        for r in refs + list(targets[i + 1 :]):
            score = similarity(t, r)
            if score >= threshold:
                hits.append((t, r, round(score, 2)))
    return hits


@dataclass
class Metrics:
    total: int
    correct: int
    confusion: dict[str, dict[str, int]]  # confusion[기대][판정]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def precision(self, label: str) -> float:
        predicted = sum(self.confusion[e][label] for e in self.confusion)
        return self.confusion[label][label] / predicted if predicted else 0.0

    def recall(self, label: str) -> float:
        actual = sum(self.confusion[label].values())
        return self.confusion[label][label] / actual if actual else 0.0

    def f1(self, label: str) -> float:
        p, r = self.precision(label), self.recall(label)
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def macro_f1(self) -> float:
        return sum(self.f1(label) for label in LABELS) / len(LABELS)


def compute_metrics(expected: Sequence[str], predicted: Sequence[str]) -> Metrics:
    """predicted에 LABELS 밖의 값(예: ERROR)이 있으면 오답으로 치고 혼동행렬 'ERROR' 열에 기록."""
    cols = list(LABELS) + sorted({p for p in predicted if p not in LABELS})
    confusion = {e: {p: 0 for p in cols} for e in LABELS}
    for e, p in zip(expected, predicted, strict=True):
        confusion[e][p] += 1
    correct = sum(e == p for e, p in zip(expected, predicted, strict=True))
    return Metrics(len(expected), correct, confusion)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def metrics_table(results: dict[str, Metrics]) -> str:
    """방식별(예: 규칙 기반 / AI) 정확도·유형별 지표 비교표."""
    names = list(results)
    lines = ["| 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines.append(
        "| **정확도** | "
        + " | ".join(f"**{pct(m.accuracy)}** ({m.correct}/{m.total})" for m in results.values())
        + " |"
    )
    lines.append(
        "| 매크로 F1 | " + " | ".join(f"{m.macro_f1:.3f}" for m in results.values()) + " |"
    )
    for label in LABELS:
        ko = LABEL_KO[label]
        lines.append(
            f"| {ko} 정밀도 / 재현율 | "
            + " | ".join(
                f"{pct(m.precision(label))} / {pct(m.recall(label))}" for m in results.values()
            )
            + " |"
        )
    return "\n".join(lines)


def confusion_table(m: Metrics) -> str:
    cols = list(next(iter(m.confusion.values())))
    head = "| 기대 \\ 판정 | " + " | ".join(LABEL_KO.get(c, c) for c in cols) + " |"
    lines = [head, "|---|" + "---|" * len(cols)]
    for e, row in m.confusion.items():
        lines.append(f"| {LABEL_KO[e]} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)
