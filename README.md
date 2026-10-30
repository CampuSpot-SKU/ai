# CampuSpot ai

> 「불편이 문제가 되기 전에」 AI 기반 캠퍼스 문제 탐지·예측·해결 서비스 **캠퍼스팟(CampuSpot)**

캠퍼스팟의 **AI 서비스**입니다. 학생 메시지가 신고인지 문의인지 가려내고, 신고는 대화로 위치·내용을 되물어 정리하며,
행정 문의는 학칙·학교 공지를 찾아 근거와 함께 답합니다. 쌓인 신고로 반복 문제를 찾고 재발 시점을 예측하는 배치도 이곳에서 돌아갑니다.

- 배포 주소: https://campuspot-frontend-890230516680.asia-northeast3.run.app (관리자 화면은 `/admin`)
- 전체 레포: https://github.com/CampuSpot-SKU

## 3개 레포가 연결되는 방식

```
[학생·관리자 브라우저]
        │
   frontend (React + TypeScript)        ← https://github.com/CampuSpot-SKU/frontend
        │  HTTPS
   backend (FastAPI)                    ← https://github.com/CampuSpot-SKU/backend
        │  HTTP (서비스 간 내부 인증 헤더)
   ai (FastAPI + Gemini, 이 레포)
```

- 이 서비스는 브라우저가 직접 부르지 않고 **backend만 호출**합니다. 반복 패턴 탐지·재발 예측·공지 수집 배치는 스케줄러가 직접 호출합니다.
- 별도의 Google Cloud Run 서비스로 배포됩니다.

## 제공하는 기능 (경로는 모두 `/api/v1` 아래)

| 경로 | 내용 |
|---|---|
| `/intent/classify` | 학생 메시지를 신고·문의·잡담으로 분류 |
| `/report/turn` | 신고 대화 에이전트 — 위치·증상을 되묻고 사진을 직접 보고 확인, 접수 내용 정리 |
| `/report/judge` | 신고 카테고리·긴급도 판정과 판정 이유 |
| `/report/say` | 학생에게 보낼 안내 문구 생성 |
| `/rag/answer`, `/rag/answer/stream` | 학칙·공지 기반 행정 문의 답변(근거 출처 포함, 실시간 스트리밍 지원) |
| `/cron/detection-scan` | 신고 데이터에서 반복 문제 후보 탐지 |
| `/cron/prediction-update` | 장소별 재발 시점 예측 갱신 |
| `/cron/crawl-notices` | 학교 공지 수집 |

## 폴더 구조

```
app/
  main.py          FastAPI 진입점
  routers/         chat.py(backend가 호출하는 경로), cron.py(배치 경로)
ai/                실제 AI 로직
  intent.py        의도 분류
  agent.py         신고 대화 에이전트(사진 포함)
  judge.py         카테고리·긴급도 판정
  priority.py      우선순위 산정
  rag.py, answer.py  검색(벡터 + 조항 번호) 및 답변 생성
  regulations.py   학칙·규정집 조항 처리
  detection.py     반복 패턴 탐지
  prediction.py    재발 예측
  notice_crawler.py, page_scraper.py, contacts.py, professors.py  학교 공지·안내 페이지·연락처 수집
  privacy.py       개인정보 마스킹
  prompts/         Gemini 프롬프트
data/              학칙·규정집 원문, 수집한 공지·안내 페이지
scripts/           임베딩 생성, 데이터 수집, 의도 분류·판정 정확도 평가
tests/             pytest 테스트와 평가용 정답 세트
docs/eval/         평가 결과
```

## 기술 스택

- Python 3.11, FastAPI, Uvicorn
- Google Gemini API(`google-genai`) — 분류·대화·답변·사진 분석, `gemini-embedding-001` 임베딩
- PostgreSQL + pgvector(SQLAlchemy) — 문서 검색·탐지·예측 데이터
- BeautifulSoup(학교 페이지 수집), Sentry(에러 추적)
- 품질 검사: ruff, mypy, pytest, gitleaks

## 배포 방식

`main` 브랜치에 push → GitHub Actions(`.github/workflows/deploy.yml`) →
비밀값 검사·린트·타입 검사·테스트 → Docker 이미지 빌드 → **Google Cloud Run** 배포.
문서 임베딩 생성과 의도 분류·판정 정확도 평가는 Actions에서 수동으로 실행합니다.
