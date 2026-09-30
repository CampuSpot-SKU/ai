"""신고 판정 규칙 기반 비교군 (S-7) — backend `app/services/slot_filling.py`의 임시 규칙(키워드)을 그대로 옮긴 것.

"AI로 바꾸면 얼마나 나아지나"를 같은 정답 세트로 비교하기 위한 용도라 규칙을 고치지 않는다.
backend의 규칙이 바뀌면 이 목록도 맞춰서 다시 옮길 것 (2026-09-30 기준 복사본).
안전 카테고리는 키워드에 걸리거나 안전 관련 긴급 표현이 있으면 긴급도 "high".
위치는 정답 세트의 "위치" 열을 그대로 쓴다 (규칙 기반도 위치 추출은 이미 잘 된다고 보고 판정만 비교).
"""
from typing import Any

CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("안전", ("미끄", "추락", "낙하", "떨어질", "붕괴", "무너", "균열", "금이 갔", "깨진 유리",
             "유리가 깨", "화재", "불이 났", "불이 난", "불이 날", "불났", "불 났", "불 난", "불길", "폭발", "연기", "가스", "비상구", "소화기", "넘어질", "다칠", "위험")),
    ("전기", ("조명", "전등", "형광등", "불이 안", "불이 꺼", "불 꺼", "불이 나갔", "불이 나가", "불 나갔", "불이 들어오", "깜빡", "콘센트", "전기",
             "누전", "감전", "정전", "스위치", "차단기", "스파크")),
    ("IT·네트워크", ("와이파이", "wifi", "인터넷", "네트워크", "프로젝터", "빔", "컴퓨터", "pc",
                    "모니터", "프린터", "출결", "랜선", "전자칠판")),
    ("청소·위생", ("냄새", "악취", "쓰레기", "더러", "청소", "벌레", "바퀴", "곰팡이", "오물",
                  "토사물", "휴지가 없", "휴지 없")),
    ("시설·설비", ("고장", "물이 새", "새요", "샌다", "누수", "수도", "정수기", "에어컨", "냉방",
                  "난방", "히터", "엘리베이터", "승강기", "변기", "막혔", "막혀", "문이 안", "손잡이", "잠금",
                  "도어락", "의자", "책상", "벤치", "파손", "부서", "깨졌", "창문", "블라인드", "배수")),
]

DEFAULT_CATEGORY = "기타"

PROBLEM_WORDS = ("안 돼", "안돼", "안 되", "안되", "안 나와", "안나와", "안 켜", "안켜", "안 열",
                 "안 닫", "안 터", "안터", "망가", "작동", "멈췄", "멈춰", "꺼져", "끊겨", "끊겨요", "없어요")

URGENT_WORDS = ("누전", "감전", "스파크", "불꽃", "화재", "불이 났", "불이 난", "불이 날", "불났", "불 났", "불 난", "불길", "폭발", "연기", "가스", "타는 냄새",
                "미끄", "추락", "낙하", "떨어질", "붕괴", "무너", "갇혔", "갇혀", "침수", "물이 넘",
                "깨진 유리", "유리가 깨", "위험", "다쳤", "다칠", "부상", "안전")

PRIVATE_PLACES = ("연구실", "사무실", "교수실", "호실", "내 방", "우리 방", "사물함", "개인")


def judge_rules(text: str, location: str | None) -> dict[str, Any]:
    lowered = text.lower()
    category = next((n for n, words in CATEGORY_KEYWORDS if any(w in lowered for w in words)), None)
    private = any(w in text for w in PRIVATE_PLACES)
    impact = "high" if (not private and bool(location)) else "low"
    urgent = category == "안전" or any(w in lowered for w in URGENT_WORDS)
    return {
        "category": category or DEFAULT_CATEGORY,
        "impact": impact,
        "urgency": "high" if urgent else "low",
        "problem_stated": category is not None or any(w in lowered for w in PROBLEM_WORDS),
    }
