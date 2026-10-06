# 학교 공지 원문 (RAG 원본 — 작업 1-14)

학교 홈페이지 공지 게시판의 공개 API에서 받은 공지입니다. `python scripts/scrape_notices.py`로 만듭니다.

| 파일 | 내용 |
|---|---|
| `notices.jsonl` | 공지 1건 = 한 줄(JSON): 제목·분류·게시일·원문 주소(`link`)·본문(마크다운) |

- 범위: 최근 1년, 8개 분류 전부(일반·장학·학사·등록납부·행사안내·대외활동·채용공지·FYP)
- 본문이 거의 없는 공지(포스터 이미지·첨부만 있는 글)도 제목으로 검색되도록 함께 저장·적재
- DB에 넣기: GitHub Actions "Embed pages" → target=notices → check → apply → search
- 새 공지만 받아 합치기: `python scripts/scrape_notices.py --incremental`
