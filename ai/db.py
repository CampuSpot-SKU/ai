"""DB 연결 — 탐지/예측/공지크롤링이 Supabase에 직접 읽고 쓸 때 사용.

- DATABASE_URL은 backend와 같은 GitHub Secret 값을 쓰는데, 설치된 드라이버는 psycopg2뿐이라
  URL이 postgresql+psycopg:// 등 어떤 형태로 들어와도 postgresql+psycopg2://로 통일한다.
  (backend/app/config.py와 같은 처리 — 안 하면 "No module named 'psycopg'"로 실패)
- 엔진은 처음 필요할 때 1번만 만든다(lazy). DATABASE_URL이 없는 CI/테스트 환경에서도
  import가 깨지지 않게 하기 위함.
- backend와 같은 Supabase Session pooler를 공유하므로 풀은 작게 (최대 5).
"""
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

_PG_SCHEMES = ("postgresql+psycopg2://", "postgresql+psycopg://", "postgresql://", "postgres://")


def normalize_database_url(url: str) -> str:
    url = url.strip()
    for scheme in _PG_SCHEMES:
        if url.startswith(scheme):
            return "postgresql+psycopg2://" + url[len(scheme):]
    return url


@lru_cache
def get_engine() -> Engine:
    url = normalize_database_url(os.environ.get("DATABASE_URL", ""))
    if not url:
        raise RuntimeError("DATABASE_URL이 설정되지 않았습니다.")
    return create_engine(url, pool_size=3, max_overflow=2, pool_pre_ping=True, pool_recycle=300)


@contextmanager
def get_session() -> Iterator[Session]:
    """사용법: `with get_session() as db: ...` — 블록이 끝나면 커밋, 예외 시 롤백."""
    session = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
