"""BigQuery adapter tests.

Most of these are error-path unit tests that run without google-cloud-bigquery
installed (or without credentials). End-to-end profile / schema-discovery
tests are gated on `BIOPSY_TEST_BIGQUERY=1` and require:

    BIOPSY_TEST_BIGQUERY=1
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    BIOPSY_TEST_BIGQUERY_TABLE=project.dataset.table
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

BIGQUERY_TESTS_ENABLED = os.environ.get("BIOPSY_TEST_BIGQUERY") == "1"


# --- URI parsing -----------------------------------------------------------


def test_bigquery_uri_parses_into_project_dataset_table() -> None:
    parsed = parse_warehouse_uri("bigquery://my-project/analytics.events")
    assert parsed is not None
    assert parsed.scheme == "bigquery"
    assert parsed.host == "my-project"
    assert parsed.path == "/analytics.events"


def test_bigquery_uri_rejects_sql_injection_in_path() -> None:
    with pytest.raises(ValueError, match="Invalid path identifier"):
        parse_warehouse_uri("bigquery://proj/ds.tbl; DROP TABLE x")


# --- adapter error paths (no credentials needed) ---------------------------


def test_missing_google_application_credentials_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No GOOGLE_APPLICATION_CREDENTIALS → MissingCredentialError before
    the driver is even imported."""
    pytest.importorskip("google.cloud.bigquery")
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    from biopsy.warehouse.bigquery import open_bigquery

    parsed = parse_warehouse_uri("bigquery://proj/ds.tbl")
    assert parsed is not None
    with pytest.raises(MissingCredentialError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        open_bigquery(None, parsed)  # type: ignore[arg-type]


def test_invalid_credentials_path_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """GOOGLE_APPLICATION_CREDENTIALS pointing nowhere fails clearly."""
    pytest.importorskip("google.cloud.bigquery")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))
    from biopsy.warehouse.bigquery import open_bigquery

    parsed = parse_warehouse_uri("bigquery://proj/ds.tbl")
    assert parsed is not None
    with pytest.raises(WarehouseDriverNotInstalledError, match="does not point to"):
        open_bigquery(None, parsed)  # type: ignore[arg-type]


def test_missing_driver_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If google-cloud-bigquery isn't installed, the error message
    points to the right pip extra."""
    # Simulate missing dependency by removing it from sys.modules and
    # blocking its re-import.
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", None)
    monkeypatch.setitem(sys.modules, "google.cloud", None)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/x.json")

    from biopsy.warehouse.bigquery import open_bigquery

    parsed = parse_warehouse_uri("bigquery://proj/ds.tbl")
    assert parsed is not None
    with pytest.raises(WarehouseDriverNotInstalledError, match=r"biopsy\[bigquery\]"):
        open_bigquery(None, parsed)  # type: ignore[arg-type]


def test_uri_without_dataset_table_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path missing `dataset.table` fails fast with a clear message."""
    pytest.importorskip("google.cloud.bigquery")
    # Point at a non-existent JSON so we don't actually hit BigQuery
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/x.json")
    from biopsy.warehouse.bigquery import _split_table

    parsed = parse_warehouse_uri("bigquery://my-project/")
    assert parsed is not None
    with pytest.raises(ValueError, match="bigquery://project/dataset.table"):
        _split_table(parsed)


# --- where-clause translation (backtick swap) ------------------------------


def test_where_clause_uses_backticks_for_bigquery() -> None:
    r"""parse_filter_expr emits `"col"` (DuckDB style); BigQuery needs backticks."""
    from biopsy.warehouse.bigquery import _build_where

    schema = {"country": "STRING", "amount": "FLOAT64"}
    clause = _build_where(["country == US", "amount > 0"], schema)
    # Identifiers swap to backticks; literals stay single-quoted.
    assert "`country`" in clause
    assert "`amount`" in clause
    assert "'US'" in clause
    assert '"country"' not in clause  # no DuckDB-style quoting remains


def test_where_clause_empty_when_no_expressions() -> None:
    from biopsy.warehouse.bigquery import _build_where

    assert _build_where([], {"foo": "STRING"}) == ""


# --- integration (gated) ---------------------------------------------------


@pytest.mark.skipif(
    not BIGQUERY_TESTS_ENABLED or not os.environ.get("BIOPSY_TEST_BIGQUERY_TABLE"),
    reason=(
        "set BIOPSY_TEST_BIGQUERY=1, GOOGLE_APPLICATION_CREDENTIALS, and "
        "BIOPSY_TEST_BIGQUERY_TABLE=project.dataset.table to run"
    ),
)
def test_bigquery_end_to_end_profile() -> None:
    from biopsy import profile

    table = os.environ["BIOPSY_TEST_BIGQUERY_TABLE"]
    project, rest = table.split(".", 1)
    uri = f"bigquery://{project}/{rest}"
    prof = profile(uri, sample=1000)
    assert prof.n_rows <= 1000
    assert prof.source_uri == uri
    assert prof.source_path is None


@pytest.mark.skipif(
    not BIGQUERY_TESTS_ENABLED or not os.environ.get("BIOPSY_TEST_BIGQUERY_TABLE"),
    reason="see test_bigquery_end_to_end_profile gating",
)
def test_bigquery_discover_schema_no_scan() -> None:
    from biopsy.warehouse.bigquery import discover_schema

    table = os.environ["BIOPSY_TEST_BIGQUERY_TABLE"]
    project, rest = table.split(".", 1)
    parsed = parse_warehouse_uri(f"bigquery://{project}/{rest}")
    assert parsed is not None
    schema, row_estimate = discover_schema(parsed)
    assert isinstance(schema, dict)
    assert len(schema) > 0
    # row_estimate may be None for views; for a regular table it should be set.
