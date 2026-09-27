import pytest

from ai.db import normalize_database_url


@pytest.mark.parametrize(
    "raw",
    [
        "postgresql+psycopg://u:p@h:5432/postgres",
        "postgresql+psycopg2://u:p@h:5432/postgres",
        "postgresql://u:p@h:5432/postgres",
        "postgres://u:p@h:5432/postgres",
        "  postgresql://u:p@h:5432/postgres\n",
    ],
)
def test_normalize_to_psycopg2(raw: str) -> None:
    assert normalize_database_url(raw) == "postgresql+psycopg2://u:p@h:5432/postgres"


def test_empty_stays_empty() -> None:
    assert normalize_database_url("") == ""
