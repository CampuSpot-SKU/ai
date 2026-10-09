"""신고 판정 유닛테스트 — Gemini를 부르지 않고 검사 규칙·재시도만 확인 (CI용)."""
import pytest

from ai import judge as judge_mod
from ai.judge import GeminiJudgement, JudgeError, JudgeRequest, judge, validate

CATS = ["전기", "시설·설비", "청소·위생", "안전", "IT·네트워크", "기타"]


def _req(text: str = "3층 화장실 물이 새요", location: str | None = "혜인관 3층 화장실") -> JudgeRequest:
    return JudgeRequest(text=text, location=location, categories=CATS)


def _g(**kw: object) -> GeminiJudgement:
    base: dict[str, object] = {
        "category": "시설·설비",
        "impact": "high",
        "urgency": "low",
        "problem_stated": True,
        "reason": "여러 학생이 쓰는 화장실이지만 당장 위험하지는 않아 보여요.",
    }
    base.update(kw)
    return GeminiJudgement.model_validate(base)


def test_validate_passes() -> None:
    out = validate(_g(), _req())
    assert out.category == "시설·설비" and out.impact == "high" and out.urgency == "low"


def test_validate_rejects_unknown_category() -> None:
    with pytest.raises(JudgeError):
        validate(_g(category="건축"), _req())


@pytest.mark.parametrize("reason", ["", "x" * 120, "P2로 봤어요.", "**위험**해요."])
def test_validate_rejects_bad_reason(reason: str) -> None:
    with pytest.raises(JudgeError):
        validate(_g(reason=reason), _req())


def test_validate_rejects_invented_place() -> None:
    with pytest.raises(JudgeError):
        validate(_g(reason="북악관 5층 복도가 위험해 보여요."), _req())


def test_judge_retries_once_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def boom(_req: JudgeRequest) -> GeminiJudgement:
        calls.append(1)
        raise RuntimeError("quota")

    monkeypatch.setattr(judge_mod, "_call_gemini", boom)
    with pytest.raises(JudgeError):
        judge(_req())
    assert len(calls) == 2


def test_judge_second_attempt_can_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    results = iter([_g(category="없는것"), _g()])
    monkeypatch.setattr(judge_mod, "_call_gemini", lambda _r: next(results))
    assert judge(_req()).category == "시설·설비"


# ---- 사진 첨부 (1-10) ----
import base64

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _photo_req(data: bytes = JPEG, mime: str = "image/jpeg") -> JudgeRequest:
    return JudgeRequest(
        text="3층 화장실 바닥에 물이 고였어요",
        location="혜인관 3층 화장실",
        categories=CATS,
        photo_base64=base64.b64encode(data).decode(),
        photo_mime=mime,  # type: ignore[arg-type]
    )


def test_photo_bytes_accepts_jpeg_and_png_and_rejects_mismatch() -> None:
    assert _photo_req().photo_bytes() == JPEG
    assert _photo_req(PNG, "image/png").photo_bytes() == PNG
    with pytest.raises(JudgeError):
        _photo_req(PNG, "image/jpeg").photo_bytes()  # 이름만 jpeg인 PNG
    with pytest.raises(JudgeError):
        _photo_req(b"hello", "image/jpeg").photo_bytes()
    assert _req().photo_bytes() is None


def test_validate_keeps_photo_note_only_when_photo_attached() -> None:
    note = "바닥에 물이 넓게 고여 있는 모습이에요."
    assert validate(_g(photo_note=note), _photo_req()).photo_note == note
    assert validate(_g(photo_note=note), _req()).photo_note is None  # 사진이 없으면 무시


@pytest.mark.parametrize("note", ["x" * 100, "P1 수준이에요.", "북악관 5층 복도 사진이에요."])
def test_validate_rejects_bad_photo_note(note: str) -> None:
    with pytest.raises(JudgeError):
        validate(_g(photo_note=note), _photo_req())


def test_contents_include_image_part_only_with_photo() -> None:
    assert len(judge_mod._contents(_req())) == 1
    parts = judge_mod._contents(_photo_req())
    assert len(parts) == 2 and parts[1].inline_data is not None
    assert parts[1].inline_data.mime_type == "image/jpeg" and parts[1].inline_data.data == JPEG


def test_judge_drops_broken_photo_and_judges_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake(req: JudgeRequest) -> GeminiJudgement:
        seen.append(req.photo_base64 is None)
        return _g()

    monkeypatch.setattr(judge_mod, "_call_gemini", fake)
    out = judge(_photo_req(b"not-an-image", "image/jpeg"))
    assert seen == [True] and out.photo_note is None
