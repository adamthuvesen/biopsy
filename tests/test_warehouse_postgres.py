"""Postgres adapter integration tests.

Gated on `BIOPSY_TEST_POSTGRES=1`. Bring up the fixture with:

    docker compose -f tests/docker-compose.postgres.yml up -d

The fixture seeds `biopsy.events` (1500 rows) and `biopsy.users` (500
rows) into a `biopsy_test` database on port 55432. Credentials are
hard-coded for the throwaway container — do not reuse the password
anywhere real.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from biopsy.warehouse import parse_warehouse_uri

# --- gating ----------------------------------------------------------------

POSTGRES_TESTS_ENABLED = os.environ.get("BIOPSY_TEST_POSTGRES") == "1"

# Default values match docker-compose.postgres.yml. Override via env if
# you're running against a different Postgres (e.g. a CI-managed instance).
PG_HOST = os.environ.get("BIOPSY_TEST_POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("BIOPSY_TEST_POSTGRES_PORT", "55432")
PG_DB = os.environ.get("BIOPSY_TEST_POSTGRES_DB", "biopsy_test")
PG_USER = os.environ.get("BIOPSY_TEST_POSTGRES_USER", "biopsy")
PG_PASSWORD = os.environ.get("BIOPSY_TEST_POSTGRES_PASSWORD", "biopsy_test")

EVENTS_URI = f"postgres://{PG_HOST}:{PG_PORT}/{PG_DB}?table=biopsy.events"
USERS_URI = f"postgres://{PG_HOST}:{PG_PORT}/{PG_DB}?table=biopsy.users"


pytestmark = pytest.mark.skipif(
    not POSTGRES_TESTS_ENABLED,
    reason=(
        "set BIOPSY_TEST_POSTGRES=1 (and start "
        "tests/docker-compose.postgres.yml) to run Postgres integration tests"
    ),
)


@pytest.fixture(autouse=True)
def _pg_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure the libpq env vars resolve to the fixture container."""
    monkeypatch.setenv("PGUSER", PG_USER)
    monkeypatch.setenv("PGPASSWORD", PG_PASSWORD)
    monkeypatch.setenv("PGHOST", PG_HOST)
    monkeypatch.setenv("PGPORT", PG_PORT)
    monkeypatch.setenv("PGDATABASE", PG_DB)


# --- end-to-end profile ----------------------------------------------------


def test_profile_postgres_table() -> None:
    """`biopsy.profile(postgres://...)` runs the full pipeline."""
    from biopsy import profile

    prof = profile(EVENTS_URI, target="converted")
    assert prof.n_rows == 1500
    # Schema: 6 columns from the fixture; biopsy may exclude the target
    # from `n_cols` accounting but `columns` should still contain everything.
    assert set(prof.columns) == {
        "event_id", "user_id", "occurred_at", "amount", "country", "converted",
    }
    assert prof.target == "converted"
    assert prof.source_uri == EVENTS_URI
    assert prof.source_path is None
    # `event_id` is unique and ID-shaped → expect at least one finding flagging it.
    id_findings = [
        f for f in prof.findings if "event_id" in f.columns and f.kind == "identifier_shape"
    ]
    assert id_findings, "expected event_id to be flagged as identifier-shaped"


def test_filter_pushdown_reduces_row_count() -> None:
    """`--filter country == US` should yield ~1/5 of the rows (one of 5 countries)."""
    from biopsy import profile

    prof = profile(EVENTS_URI, target="converted", where=["country == US"])
    # Round-trip the fixture math: g % 5 + 1 = 1 → country = 'US' for g in 0,5,10,...
    # That's exactly 1500 / 5 = 300 rows.
    assert prof.n_rows == 300


def test_sample_becomes_limit_on_postgres() -> None:
    """`--sample N` translates to LIMIT N, not local reservoir sampling."""
    from biopsy import profile

    prof = profile(EVENTS_URI, sample=100)
    assert prof.n_rows == 100


# --- schema discovery (cheap path, no row data) ----------------------------


def test_discover_schema_returns_columns_and_estimate() -> None:
    """`discover_schema` hits INFORMATION_SCHEMA + pg_class — no row scan."""
    from biopsy.warehouse.postgres import discover_schema

    parsed = parse_warehouse_uri(EVENTS_URI)
    assert parsed is not None
    con = duckdb.connect(":memory:")
    schema, est = discover_schema(con, parsed)
    assert set(schema) == {
        "event_id", "user_id", "occurred_at", "amount", "country", "converted",
    }
    # reltuples is the planner estimate. After ANALYZE on a 1500-row table
    # it should be within an order of magnitude.
    assert est is not None
    assert 500 < est < 5000


def test_doctor_against_postgres_uri_uses_cheap_path() -> None:
    """`biopsy doctor` against a postgres URI should not pull row data."""
    from typer.testing import CliRunner

    from biopsy.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", EVENTS_URI])
    assert result.exit_code == 0, result.output
    # Schema-only mode tip appears in stderr (rich Console).
    assert "schema-only" in result.output or "rows (estimate)" in result.output
    # Column names from the fixture should appear in the table output.
    assert "event_id" in result.output
    assert "country" in result.output


# --- read-only enforcement (runtime check) ---------------------------------


def test_attached_postgres_rejects_insert() -> None:
    """READ_ONLY at attach + session must make writes fail server-side."""
    from biopsy.warehouse.postgres import open_postgres

    parsed = parse_warehouse_uri(EVENTS_URI)
    assert parsed is not None
    con = duckdb.connect(":memory:")
    result = open_postgres(con, parsed)
    assert result.scan_sql is not None
    alias = result.scan_sql.split(".", 1)[0]

    # DuckDB's postgres extension surfaces remote errors as duckdb.Error /
    # IOException. Either way, an INSERT against the attached read-only
    # database must fail — that's the contract.
    with pytest.raises(Exception) as exc_info:
        con.execute(
            f"INSERT INTO {alias}.biopsy.events "
            f"VALUES (99999, 1, NOW(), 1.0, 'US', false)"
        )
    msg = str(exc_info.value).lower()
    assert any(
        keyword in msg
        for keyword in ("read-only", "read only", "readonly", "cannot")
    ), f"expected read-only rejection, got: {exc_info.value}"

    if result.cleanup is not None:
        result.cleanup()


# --- mixed-source compare (smoke) ------------------------------------------


def test_compare_postgres_to_postgres(tmp_path: Path) -> None:
    """Two Postgres profiles compare cleanly — exercises the same-source path."""
    from biopsy import compare_profiles, profile

    prof_a = profile(EVENTS_URI, target="converted", where=["country == US"])
    prof_b = profile(EVENTS_URI, target="converted", where=["country == GB"])
    report = compare_profiles(prof_a, prof_b)
    # Same schema, different countries → no schema diff, target rate may differ.
    assert report.schema.added == []
    assert report.schema.removed == []
    assert isinstance(report.drifts, list)
