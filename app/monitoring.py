"""에러 트래킹(Sentry) — Gemini 호출 실패를 개발팀에 알림 (S-3·1-15, 명세 6장).

- SENTRY_DSN 환경변수가 없으면 아무것도 하지 않음(로컬·테스트).
- 엔드포인트가 Gemini 실패를 잡아 503으로 바꾸면 Sentry가 자동으로 못 보므로, 잡는 자리에서 report_failure로 직접 보냄.
  태그 `ai_failure=<기능>`으로 묶어 Sentry 알림 규칙(새 이슈·빈도)을 걸 수 있다.
- 개인정보: send_default_pii=False(요청 본문·IP를 안 보냄), include_local_variables=False(스택의 지역 변수를 안 보냄). 예외 메시지에는 학생 문장이 들어가지 않게 각 모듈이 짧은 원인만 담음.
"""
import os

import sentry_sdk


def init_sentry() -> None:
    dsn = os.environ.get("SENTRY_DSN", "")
    if not dsn:
        return
    sentry_sdk.init(
        dsn=dsn,
        send_default_pii=False,
        include_local_variables=False,  # 스택의 지역 변수(학생 문장이 든 요청 등)를 보내지 않음
        traces_sample_rate=0.2,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
    )


def report_failure(feature: str, error: BaseException) -> None:
    """잡아서 처리한 Gemini·AI 실패를 Sentry 이벤트로 남김 (SENTRY_DSN이 없으면 아무 일도 안 함)."""
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("ai_failure", feature)
        sentry_sdk.capture_exception(error)
