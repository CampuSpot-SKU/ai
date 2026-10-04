# 홈페이지 안내 페이지 수집 보고서

> `python scripts/scrape_pages.py`가 자동으로 만듭니다. 직접 고치지 마세요.

요약: ok 47, short 2, duplicate 3

| slug | 제목 | 상태 | 글자 수 | 제목 수 | 표 | 출처 |
|---|---|---|---|---|---|---|
| academic-calendar | 학부 학사일정 (2026) | ok | 1627 | 0 | 0 | manual |
| academic-status | 학사경고·자퇴·제적 | ok | 843 | 5 | 0 | web |
| academic-warning | 학사경고 | ok | 400 | 2 | 0 | web |
| all-faq | 전체 FAQ | ok | 19008 | 43 | 1 | web |
| campus-life | 증명서 | ok | 1176 | 1 | 1 | web |
| campus-network | 유·무선 인터넷 | ok | 1103 | 11 | 1 | web |
| change-major | 전과 | ok | 460 | 7 | 0 | web |
| change-major-and-readmission | 전과·재입학 | ok | 729 | 14 | 0 | web |
| course-evaluations | 강의평가 | ok | 751 | 8 | 0 | web |
| course-registration | 수강 | ok | 1382 | 8 | 1 | web |
| curriculum | 교육과정 | ok | 5100 | 6 | 4 | web |
| department-sites | 학과·학부 홈페이지와 교수진 페이지 안내 | ok | 3139 | 0 | 0 | manual |
| directions | 찾아오시는길 | ok | 1101 | 9 | 6 | web |
| domestic-exchange | 국내교류 | ok | 750 | 8 | 0 | web |
| dropout-expulsion | 자퇴·제적 | ok | 443 | 3 | 0 | web |
| exam-and-grade | 시험·성적 | ok | 3044 | 17 | 2 | web |
| general-english | 교양영어 | ok | 345 | 1 | 0 | web |
| graduation | 졸업·수료 | ok | 972 | 9 | 1 | web |
| health-center | 보건진료소 | ok | 1518 | 18 | 3 | web |
| human-rights | 인권센터 | ok | 1281 | 15 | 0 | web |
| interchange_student_system | 국제교류학생제도 | ok | 1822 | 7 | 0 | web |
| internal-scholarships | 교내장학 | ok | 2584 | 15 | 1 | web |
| international-exchange | 국제교류 | short | 59 | 0 | 0 | web |
| issuance | 증명서 | ok | 1176 | 1 | 1 | web |
| it-services | 웹메일 | ok | 381 | 2 | 0 | web |
| leave | 휴학 | ok | 858 | 7 | 0 | web |
| leave-and-return | 휴학·복학 | ok | 1376 | 9 | 0 | web |
| library | 학술정보관 | ok | 7601 | 38 | 7 | web |
| micro-degree-major | 마이크로디그리전공 | ok | 1990 | 5 | 1 | web |
| microdegree-major-guide | 마이크로디그리 전공 안내 | ok | 415 | 5 | 0 | web |
| microdegree-major-qna | Q&A | ok | 808 | 0 | 0 | web |
| microdegree-programs | 마이크로디그리 과정 소개 | ok | 767 | 0 | 1 | web |
| military | 병무 | ok | 3879 | 35 | 4 | web |
| military-service | 병무·예비군 | ok | 6235 | 49 | 6 | web |
| minor-double-major | 부전공·복수전공 | ok | 401 | 3 | 0 | web |
| organization-phone | 서경대학교 조직도·부서 연락처 | ok | 12546 | 0 | 0 | manual |
| partner_institutions_abroad | 국제교류학교 | ok | 6279 | 37 | 0 | web |
| professors | 학과 교수진 연락처 | ok | 23711 | 0 | 0 | manual |
| reinstatement | 재입학 | short | 269 | 7 | 0 | web |
| reserve-forces | 병무·예비군 | duplicate | 6235 | 49 | 6 | web |
| return | 복학 | ok | 518 | 2 | 0 | web |
| scholarship-regulations | 장학금지급규정 | ok | 1322 | 8 | 0 | web |
| seasonal-sessions | 계절학기 | ok | 534 | 7 | 0 | web |
| security | 보안 | ok | 673 | 3 | 0 | web |
| software | 소프트웨어 | ok | 2039 | 13 | 1 | web |
| student-activities | 학생자치기구 | duplicate | 1817 | 0 | 0 | web |
| student-clubs | 동아리 | ok | 3074 | 6 | 0 | web |
| student-id | 학생증 | ok | 430 | 6 | 0 | web |
| student-insurance | 대학안전사고보상공제 | ok | 1006 | 8 | 2 | web |
| student-union | 학생자치기구 | ok | 1817 | 0 | 0 | web |
| volunteer | 사회봉사 | ok | 1212 | 12 | 3 | web |
| webmail | 웹메일 | duplicate | 381 | 2 | 0 | web |

## 사람이 확인·입력할 페이지

- `international-exchange` (short) https://www.skuniv.ac.kr/international-exchange — 본문이 짧음 — 사이트 화면과 비교해 빠진 내용이 없는지 확인 → `data/pages/manual/international-exchange.md`에 복붙하거나 스크린샷을 주세요
- `reinstatement` (short) https://www.skuniv.ac.kr/reinstatement — 본문이 짧음 — 사이트 화면과 비교해 빠진 내용이 없는지 확인 → `data/pages/manual/reinstatement.md`에 복붙하거나 스크린샷을 주세요

## 중복 (저장 안 함)

- `reserve-forces`: military-service과 같은 내용 — 저장하지 않음
- `student-activities`: student-union과 같은 내용 — 저장하지 않음
- `webmail`: it-services과 같은 내용 — 저장하지 않음
