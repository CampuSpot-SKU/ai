# 학칙 원본 (1-4b)

`hakchik-YYYYMMDD.txt` = 학교 학칙 PDF를 `pdftotext -layout`으로 뽑은 글 (파일명 날짜 = 시행일).
PDF 자체는 올리지 않는다(출처: 학교 홈페이지 규정 페이지, 공개 자료).

- 현재: `hakchik-20251001.txt` (2025.10.1 시행 학칙, 조 122개 중 삭제 10개 제외 112개 적재)
- 파서: `ai/regulations.py` (제N조 단위, 장·절·조의2 처리, 부칙·별표·서식 제외)
- 적재: Actions → "Embed pages" → target = regulations (check → apply → search)

## 학칙이 개정되면 (사람이 하는 일)
1. 새 학칙 PDF를 받아 `pdftotext -layout 파일.pdf hakchik-새시행일.txt`로 변환해 이 폴더에 넣는다.
2. push 후 Actions에서 target = regulations, action = apply — **바뀐 조만** 다시 임베딩된다(폴더에서 날짜가 가장 늦은 파일을 쓴다).
3. 새 파일이 시행일이 다른 새 판이면 옛 파일은 지워도 된다. 옛 조가 사라졌다면 `remove_missing`를 켜고 apply.

개정 사전공고·공포는 학교 공지로 올라오므로 답변 때 공지 검색으로 함께 안내한다(1-4c).

## 규정집 (1-4d)
`gyujeongjip-YYYYMMDD.txt` = 학교 규정집 PDF(학칙 외 230여 개 규정)를 `pdftotext -layout`으로 뽑은 글 (파일명 날짜 = 규정집 기준일).

- 현재: `gyujeongjip-20240901.txt` (2024.9.1 기준, 규정 229개 / 조 3,456개 적재, 삭제 조 41개 제외)
- 학칙 본문은 위 `hakchik-*.txt`로 따로 적재하므로 규정집 안의 "서경대학교 학칙"은 건너뛴다.
- 모든 청크 머리말에 `2024.9.1 기준 규정집`이 붙고, 답변 칩에는 `2024.9.1 기준`이 표시된다 — 규정은 공지로 일부만 개정되기 때문.
- 파서: `ai/regulations.py`의 `parse_book` (쪽 머리글·바닥글 제거 → 규정명 인식 → 규정별 조 단위로 분리)
- 적재: Actions → "Embed pages" → target = regulations (check → apply → search). 학칙과 규정집이 함께 처리된다.
- `제N조` 직접 검색: 질문에 규정 이름이 있으면 그 규정에서, 없으면 학칙 본문에서만 찾는다.

### 규정집이 새로 나오면
1. 새 PDF를 `pdftotext -layout`으로 변환해 `gyujeongjip-새기준일.txt`로 넣는다(폴더에서 날짜가 가장 늦은 파일을 사용).
2. push 후 target = regulations, action = apply — 바뀐 조만 다시 임베딩. 사라진 조는 `remove_missing`를 켜고 apply.
