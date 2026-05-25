"""Schema-only discovery for `biopsy doctor` on warehouse URIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import duckdb

from biopsy.io import kind_of
from biopsy.stats import ColumnStats
from biopsy.warehouse import parse_warehouse_uri

if TYPE_CHECKING:
    from biopsy.warehouse._base import ParsedURI

SchemaDiscoverer = Callable[
    ["ParsedURI", str | None],
    tuple[dict[str, str], int | None],
]


def _object_store_schema(
    parsed: ParsedURI,
    credentials_env: str | None,
) -> tuple[dict[str, str], int | None]:
    from biopsy.warehouse.object_store import discover_schema

    con = duckdb.connect(":memory:")
    try:
        return discover_schema(con, parsed), None
    finally:
        con.close()


def _postgres_schema(
    parsed: ParsedURI,
    credentials_env: str | None,
) -> tuple[dict[str, str], int | None]:
    from biopsy.warehouse.postgres import discover_schema

    con = duckdb.connect(":memory:")
    try:
        return discover_schema(con, parsed, credentials_env=credentials_env)
    finally:
        con.close()


def _bigquery_schema(
    parsed: ParsedURI,
    credentials_env: str | None,
) -> tuple[dict[str, str], int | None]:
    from biopsy.warehouse.bigquery import discover_schema

    return discover_schema(parsed, credentials_env=credentials_env)


def _snowflake_schema(
    parsed: ParsedURI,
    credentials_env: str | None,
) -> tuple[dict[str, str], int | None]:
    from biopsy.warehouse.snowflake import discover_schema

    return discover_schema(parsed, credentials_env=credentials_env)


_SCHEMA_REGISTRY: dict[str, SchemaDiscoverer] = {
    "s3": _object_store_schema,
    "s3a": _object_store_schema,
    "https": _object_store_schema,
    "http": _object_store_schema,
    "gs": _object_store_schema,
    "gcs": _object_store_schema,
    "postgres": _postgres_schema,
    "postgresql": _postgres_schema,
    "bigquery": _bigquery_schema,
    "snowflake": _snowflake_schema,
}


def discover_warehouse_schema(
    path: str,
    *,
    credentials_env: str | None,
) -> tuple[dict[str, ColumnStats], str, int | None] | None:
    """Return doctor stats tuple for warehouse URIs, or None if not applicable."""
    parsed = parse_warehouse_uri(path)
    if parsed is None:
        return None
    discover = _SCHEMA_REGISTRY.get(parsed.scheme)
    if discover is None:
        return None
    schema, row_estimate = discover(parsed, credentials_env)
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
