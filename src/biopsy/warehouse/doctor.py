"""Schema-only discovery for `biopsy doctor` on warehouse URIs."""

from __future__ import annotations

import duckdb

from biopsy.io import kind_of
from biopsy.stats import ColumnStats
from biopsy.warehouse import parse_warehouse_uri

_OBJECT_STORE_SCHEMES = {"s3", "s3a", "https", "http", "gs", "gcs"}


def discover_warehouse_schema(
    path: str,
    *,
    credentials_env: str | None,
) -> tuple[dict[str, ColumnStats], str, int | None] | None:
    """Return doctor stats tuple for warehouse URIs, or None if not applicable."""
    parsed = parse_warehouse_uri(path)
    if parsed is None:
        return None
    scheme = parsed.scheme

    if scheme in _OBJECT_STORE_SCHEMES:
        from biopsy.warehouse.object_store import discover_schema

        con = duckdb.connect(":memory:")
        try:
            schema, row_estimate = discover_schema(con, parsed), None
        finally:
            con.close()
    elif scheme in {"postgres", "postgresql"}:
        from biopsy.warehouse.postgres import discover_schema

        con = duckdb.connect(":memory:")
        try:
            schema, row_estimate = discover_schema(con, parsed, credentials_env=credentials_env)
        finally:
            con.close()
    elif scheme == "bigquery":
        from biopsy.warehouse.bigquery import discover_schema

        schema, row_estimate = discover_schema(parsed, credentials_env=credentials_env)
    elif scheme == "snowflake":
        from biopsy.warehouse.snowflake import discover_schema

        schema, row_estimate = discover_schema(parsed, credentials_env=credentials_env)
    else:
        return None

    return stats_from_schema(schema, row_estimate, parsed.qualified)


def stats_from_schema(
    schema: dict[str, str],
    row_estimate: int | None,
    qualified: str,
) -> tuple[dict[str, ColumnStats], str, int | None]:
    stats: dict[str, ColumnStats] = {
        name: ColumnStats(
            name=name,
            dtype=dtype,
            kind=kind_of(dtype),
            n=0,
            n_null=0,
            n_unique=0,
            null_rate=0.0,
        )
        for name, dtype in schema.items()
    }
    display = qualified.rsplit("/", 1)[-1] or qualified
    return stats, display, row_estimate
