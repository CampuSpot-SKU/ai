# ai

캠퍼스팟 AI 서비스 — 의도분류 / RAG / 우선순위 산정 / 반복패턴 탐지 / 재발예측.

**별도 Cloud Run 서비스로 배포됨** (`campuspot-ai`). backend는 이 서비스를
HTTP로 호출한다 (`AI_SERVICE_URL` + `X-Internal-Secret: AI_SERVICE_SECRET` 헤더).

- `app/` — FastAPI 진입점, 라우터 (backend가 호출하는 엔드포인트 + Cloud Scheduler가
  직접 호출하는 배치 엔드포인트)
- `ai/` — 실제 AI 로직 모듈 (intent/rag/priority/detection/prediction/notice_crawler),
  `app/` 라우터에서 import해서 사용

로컬에서 패키지만 따로 쓰고 싶으면(예: 노트북에서 실험) `pip install -e .`로 설치 가능.

자세한 배경은 [backend 레포의 campus-esm-chatbot-spec.md](https://github.com/CampuSpot-SKU/backend/blob/main/docs/campus-esm-chatbot-spec.md) 8장 참고.
