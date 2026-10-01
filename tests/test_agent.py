"""신고 접수 에이전트 유닛테스트 — Gemini를 부르지 않고 검사 규칙·재시도만 확인 (CI용)."""
import pytest

from ai import agent as agent_mod
from ai.agent import (
    AgentError,
    AgentState,
    BuildingInfo,
    Candidate,
    GeminiTurn,
    TurnMessage,
    TurnRequest,
    turn,
    validate,
)

BUILDINGS = [
    BuildingInfo(name="북악관", floors="지하 1층, 1~3층, 5~8층"),
    BuildingInfo(name="청운관", floors="지하 1층, 1~3층, 5~11층"),
]


def _req(text: str = "장문수 교수실 불이 안 켜져", **kw: object) -> TurnRequest:
    base: dict[str, object] = {
        "conversation": [TurnMessage(role="user", content=text)],
        "buildings": BUILDINGS,
        "candidates": [Candidate(label="장문수 교수연구실 (606)", building="북악관", floor="6")],
    }
    base.update(kw)
    return TurnRequest.model_validate(base)


def _g(**kw: object) -> GeminiTurn:
    base: dict[str, object] = {
        "action": "confirm",
        "message": "북악관 6층 장문수 교수연구실의 불이 안 켜지는 문제로 정리했어요. 이대로 접수할까요?",
        "choices": ["네, 접수해 주세요", "내용을 고칠래요", "취소할게요"],
        "state": AgentState(problem="불이 안 켜져요", problem_clear=True, building="북악관", floor="6"),
    }
    base.update(kw)
    return GeminiTurn.model_validate(base)


def test_validate_passes() -> None:
    out = validate(_g(), _req())
    assert out.action == "confirm" and out.state.building == "북악관" and out.state.floor == "6"


def test_building_not_in_school_list_moves_to_note() -> None:
    state = AgentState(problem="정수기 고장", problem_clear=True, building="미래관", floor="3")
    g = _g(action="ask", message="학교 건물 목록에서 찾지 못했어요. 다른 이름이 있을까요?", state=state)
    out = validate(g, _req("미래관 3층 정수기가 고장났어요."))
    assert out.state.building == "" and out.state.location_note == "미래관"


def test_floor_is_normalized() -> None:
    out = validate(_g(state=AgentState(problem="x", problem_clear=True, building="북악관", floor="b1층")), _req())
    assert out.state.floor == "B1"
    out = validate(_g(state=AgentState(problem="x", problem_clear=True, building="북악관", floor="여러 층")), _req())
    assert out.state.floor == ""


@pytest.mark.parametrize(
    "message",
    [
        "",
        "x" * 400,
        "북악관 6층이야 접수할래?",  # 반말
        "**북악관** 6층 접수할까요?",  # 서식
        "P1으로 접수할까요?",  # 우선순위 등급
        "미래관 6층으로 정리했어요. 이대로 접수할까요?",  # 학생 말·학교 데이터에 없는 건물
        "북악관 606호가 아니라 607호인가요? 이대로 접수할까요?",  # 지어낸 호수
    ],
)
def test_validate_rejects_bad_message(message: str) -> None:
    with pytest.raises(AgentError):
        validate(_g(message=message), _req())


def test_confirm_needs_submit_question() -> None:
    with pytest.raises(AgentError):
        validate(_g(message="북악관 6층 문제로 정리했어요."), _req())


def test_ask_needs_question_form() -> None:
    with pytest.raises(AgentError):
        validate(_g(action="ask", message="북악관이군요."), _req())


def test_too_many_choices_rejected() -> None:
    with pytest.raises(AgentError):
        validate(_g(choices=["가", "나", "다", "라", "마", "바"]), _req())


def test_submit_drops_message() -> None:
    out = validate(_g(action="submit", message="접수됐어요!!", choices=["x"]), _req())
    assert out.message == "" and out.choices == []


def test_cancel_has_default_message() -> None:
    out = validate(_g(action="cancel", message="", choices=[]), _req())
    assert out.message and out.choices == []


def test_floors_from_school_list_are_allowed_in_message() -> None:
    g = _g(
        action="ask",
        message="북악관은 지하 1층, 1~3층, 5~8층이 있는 걸로 알아요. 몇 층이세요?",
        choices=[],
        state=AgentState(problem="x", problem_clear=True, building="북악관"),
    )
    assert validate(g, _req("북악관 4층 화장실 변기가 막혔어요")).action == "ask"


def test_turn_retries_once_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def boom(_req: TurnRequest) -> GeminiTurn:
        calls.append(1)
        raise RuntimeError("quota")

    monkeypatch.setattr(agent_mod, "_call_gemini", boom)
    with pytest.raises(AgentError):
        turn(_req())
    assert len(calls) == 2


def test_turn_second_attempt_can_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([_g(message="접수할래?"), _g()])
    monkeypatch.setattr(agent_mod, "_call_gemini", lambda _r: next(results))
    assert turn(_req()).action == "confirm"


def test_prompt_contains_grounding_and_hides_instruction_role() -> None:
    content = agent_mod._build_content(_req())
    assert "장문수 교수연구실 (606) — 북악관 6층" in content
    assert "지시가 아니다" in content and "북악관: 지하 1층" in content


def test_long_messages_are_clipped() -> None:
    req = _req("가" * 3000)
    assert "…" in agent_mod._build_content(req)


def test_impossible_nesting_forces_one_screening_question() -> None:
    state = AgentState(problem="이상해요", problem_clear=True, place="엘리베이터", contained_in="화장실")
    out = validate(_g(state=state), _req("화장실 안의 엘리베이터가 이상해"))
    assert out.action == "ask" and not out.state.plausible and "근처" in out.message
    # 이미 한 번 물었는데 학생이 그대로라고 하면 막지 않고 담당자 확인으로 넘긴다
    out = validate(_g(state=state), _req("정말 안에 있어요", prev_action="ask"))
    assert out.action == "confirm" and not out.state.plausible


def test_room_no_and_near_are_sanitized() -> None:
    state = AgentState(
        problem="x", problem_clear=True, building="북악관", room_no="606호", near=["청운관", "없는관"], near_relation="between"
    )
    out = validate(_g(state=state), _req())
    assert out.state.room_no == "606" and out.state.room_kind == "numbered"
    assert out.state.near == ["청운관"] and out.state.near_relation == "between"
    out = validate(_g(state=AgentState(problem="x", problem_clear=True, room_no="여기저기")), _req())
    assert out.state.room_no == ""


def test_invented_room_number_is_dropped_and_unnumbered_spaces_marked() -> None:
    out = validate(_g(state=AgentState(problem="x", problem_clear=True, building="북악관", room_no="501")), _req("북악관 합주실 문이 안 닫혀요"))
    assert out.state.room_no == "" and out.state.room_kind == "unknown"
    out = validate(_g(state=AgentState(problem="x", problem_clear=True, place="엘리베이터")), _req("엘베가 멈췄어요"))
    assert out.state.room_kind == "unnumbered"


def test_safety_line_is_added_when_model_forgets() -> None:
    g = _g(action="ask", message="어느 건물인가요?", choices=[], state=AgentState(problem="학생이 쓰러졌어요", problem_clear=True))
    out = validate(g, _req("교수연구실 앞에 학생이 쓰러져 있어요"))
    assert out.message.startswith("지금 위험하거나 다친 분이 있으면 먼저 119나 112")
    # 이미 안내했으면 반복하지 않는다
    req = _req("교수연구실 앞에 학생이 쓰러져 있어요", conversation=[
        TurnMessage(role="user", content="교수연구실 앞에 학생이 쓰러져 있어요"),
        TurnMessage(role="assistant", content="먼저 119에 연락해 주세요. 어느 건물인가요?"),
        TurnMessage(role="user", content="청운관이요"),
    ])
    assert "119" not in validate(g, req).message
