"""규칙 기반 의도 분류기 — S-7a 정량 평가의 비교군 (명세서 10-4). 배포 코드에는 쓰지 않음.

"AI를 안 쓰고 키워드·정규식으로 만들면 어느 정도 나오나"를 재기 위한 기준선.
허수아비가 되지 않도록 개발 세트(tests/golden/intent_classification.csv)만 보고 합리적으로 작성했고,
평가 세트(intent_eval_holdout.csv)를 보고 규칙을 고치지 않는다 (튜닝 금지 규칙은 AI와 동일).

판정 순서
1. 직전 챗봇 메시지가 신고 정보(위치·층·상황)를 되물었으면 → report
2. 직전 챗봇 메시지가 "신고로 접수할까요, 안내가 필요하신가요?"였으면 → 답변 단어로 판정
3. 발화에서 "문제 상황" 단어와 "제도·절차 문의" 단어를 찾음
   - 문제 단어가 있으면서 문의 단어나 "어떻게 해야/누구한테" 같은 방법 표현도 있으면 → unclear
   - 문제만 → report, 문의만 → inquiry
   - 둘 다 없으면 이전 대화의 마지막 학생 발화 판정을 이어받고, 그것도 없으면 → unclear
"""

import re
from collections.abc import Sequence

Label = str  # "report" | "inquiry" | "unclear"

# 단어 목록은 개발 세트 34문장에 나온 표현과 그 활용형만 사용 (평가 세트 단어를 넣으면 비교가 불공정해짐)
# 문제 상황 (고장·파손·누수·오염·IT 장애)
PROBLEM = re.compile(
    r"고장|부서|새[요고는]|깜빡|냄새|먹었|끊[겨김어기]|막[혔힘히]|나갔|미끄|약한|"
    r"안\s?(켜|되|돼|들어가)"
)
# 제도·절차 문의
INQUIRY = re.compile(
    r"신청|기간|언제|어디서|어디로|어떻게|조건|발급|증명서|장학|등록금|휴학|졸업|수강|"
    r"학점|전공|성적|문의|뽑을|내요|분실물"
)
# 문제 + 방법을 같이 묻는 표현 (신고인지 문의인지 섞인 경우)
MIXED = re.compile(r"어떻게 해야|어떻게 하죠|누구한테|수 있나요")

ASK_SLOT = re.compile(r"위치|몇 층|어느 건물|상황|자세히")
ASK_REPORT_OR_INQUIRY = re.compile(r"신고로 접수")
WANT_REPORT = re.compile(r"신고|접수")
WANT_INQUIRY = re.compile(r"안내")


def _classify_text(text: str) -> Label | None:
    """발화만 보고 판정. 단서가 전혀 없으면 None."""
    has_problem = bool(PROBLEM.search(text))
    has_inquiry = bool(INQUIRY.search(text))
    if has_problem and (has_inquiry or MIXED.search(text)):
        return "unclear"
    if has_problem:
        return "report"
    if has_inquiry:
        return "inquiry"
    if MIXED.search(text):
        return "unclear"
    return None


def classify_rules(text: str, history: Sequence[tuple[str, str]] = ()) -> Label:
    """history: (role, content) 목록 — role은 "user" 또는 "assistant"."""
    last_bot = next((c for r, c in reversed(history) if r == "assistant"), "")
    if last_bot and ASK_REPORT_OR_INQUIRY.search(last_bot):
        if WANT_REPORT.search(text):
            return "report"
        if WANT_INQUIRY.search(text):
            return "inquiry"
        return "unclear"
    if last_bot and ASK_SLOT.search(last_bot):
        return "report"

    label = _classify_text(text)
    if label is not None:
        return label
    last_user = next((c for r, c in reversed(history) if r == "user"), "")
    if last_user:
        return _classify_text(last_user) or "unclear"
    return "unclear"
