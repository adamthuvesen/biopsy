"""Snowflake adapter tests.

Error-path unit tests run without snowflake-connector-python or credentials.
End-to-end tests are gated on `BIOPSY_TEST_SNOWFLAKE=1` and require:

    BIOPSY_TEST_SNOWFLAKE=1
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER
    SNOWFLAKE_PRIVATE_KEY_PATH (or SNOWFLAKE_PASSWORD)
    SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
    BIOPSY_TEST_SNOWFLAKE_TABLE=db.schema.table
"""

from __future__ import annotations

import os
import sys

import pytest

from biopsy.warehouse import (
    MissingCredentialError,
    WarehouseDriverNotInstalledError,
    parse_warehouse_uri,
)

SNOWFLAKE_TESTS_ENABLED = os.environ.get("BIOPSY_TEST_SNOWFLAKE") == "1"


# --- URI parsing -----------------------------------------------------------


def test_snowflake_uri_parses_three_part_path() -> None:
    parsed = parse_warehouse_uri("snowflake://my-acct/SALES.PUBLIC.ORDERS")
    assert parsed is not None
    assert parsed.scheme == "snowflake"
    assert parsed.host == "my-acct"
    assert parsed.path == "/SALES.PUBLIC.ORDERS"


def test_snowflake_uri_rejects_sql_injection() -> None:
    with pytest.raises(ValueError, match="Invalid path identifier"):
        parse_warehouse_uri("snowflake://acct/db.schema.tbl; DROP TABLE x")


# --- adapter error paths (no credentials needed) ---------------------------


def test_missing_required_creds_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """SNOWFLAKE_ACCOUNT unset → MissingCredentialError before connect."""
    pytest.importorskip("snowflake.connector")
    for var in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"):
        monkeypatch.delenv(var, raising=False)
    from biopsy.warehouse.snowflake import open_snowflake

    parsed = parse_warehouse_uri("snowflake://acct/db.schema.tbl")
    assert parsed is not None
    with pytest.raises(MissingCredentialError, match="SNOWFLAKE_ACCOUNT"):
        open_snowflake(None, parsed)  # type: ignore[arg-type]


def test_missing_auth_method_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required creds set but no PASSWORD or PRIVATE_KEY_PATH → typed error."""
    pytest.importorskip("snowflake.connector")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test-acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "test-user")
    for var in ("SNOWFLAKE_PRIVATE_KEY_PATH", "SNOWFLAKE_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    from biopsy.warehouse.snowflake import open_snowflake

    parsed = parse_warehouse_uri("snowflake://acct/db.schema.tbl")
    assert parsed is not None
    with pytest.raises(
        WarehouseDriverNotInstalledError,
        match="PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD",
    ):
        open_snowflake(None, parsed)  # type: ignore[arg-type]


def test_missing_driver_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If snowflake-connector-python isn't installed, error points to extra."""
    monkeypatch.setitem(sys.modules, "snowflake.connector", None)
    monkeypatch.setitem(sys.modules, "snowflake", None)
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test-acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "test-user")

    from biopsy.warehouse.snowflake import open_snowflake

    parsed = parse_warehouse_uri("snowflake://acct/db.schema.tbl")
    assert parsed is not None
    with pytest.raises(WarehouseDriverNotInstalledError, match=r"biopsy\[snowflake\]"):
        open_snowflake(None, parsed)  # type: ignore[arg-type]


def test_uri_without_three_part_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("snowflake.connector")
    from biopsy.warehouse.snowflake import _split_table

    parsed = parse_warehouse_uri("snowflake://acct/SALES")
    assert parsed is not None
    with pytest.raises(ValueError, match="snowflake://account/db.schema.table"):
        _split_table(parsed)


# --- where-clause (no quote translation needed for Snowflake) --------------


def test_where_clause_uses_double_quotes() -> None:
    """Snowflake uses DuckDB-style double-quoted identifiers."""
    from biopsy.warehouse.snowflake import _build_where

    schema = {"COUNTRY": "TEXT", "AMOUNT": "NUMBER"}
    clause = _build_where(["COUNTRY == US", "AMOUNT > 0"], schema)
    assert '"COUNTRY"' in clause
    assert '"AMOUNT"' in clause
    assert "'US'" in clause


# --- integration (gated) ---------------------------------------------------


@pytest.mark.skipif(
    not SNOWFLAKE_TESTS_ENABLED or not os.environ.get("BIOPSY_TEST_SNOWFLAKE_TABLE"),
    reason=(
        "set BIOPSY_TEST_SNOWFLAKE=1, SNOWFLAKE_* creds, and "
        "BIOPSY_TEST_SNOWFLAKE_TABLE=db.schema.table to run"
    ),
)
def test_snowflake_end_to_end_profile() -> None:
    from biopsy import profile

    table = os.environ["BIOPSY_TEST_SNOWFLAKE_TABLE"]
    account = os.environ["SNOWFLAKE_ACCOUNT"]
    uri = f"snowflake://{account}/{table}"
    prof = profile(uri, sample=1000)
    assert prof.n_rows <= 1000
    assert prof.source_uri == uri
    assert prof.source_path is None
