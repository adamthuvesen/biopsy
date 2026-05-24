"""BigQuery adapter via google-cloud-bigquery → Arrow → register with DuckDB.

There's no production-grade DuckDB BigQuery extension, so we use the
official Python client to execute a remote `SELECT` and fetch results as
Arrow. The Arrow table is registered with biopsy's local DuckDB and
profiled like any other dataset.

Push-down is mandatory for warehouse sources — otherwise we'd transfer the
whole table over the wire. This adapter pushes both `--filter` predicates
and `--sample N` (as `LIMIT N`) into the remote `SELECT`. Filter
expressions are parsed with biopsy's standard `parse_filter_expr` and
identifier-quoted with backticks (BigQuery convention).

A dry-run estimate is reported before the real query: BigQuery returns the
number of bytes the SELECT will scan. We warn above 5 GB but don't block —
the user explicitly opted into the source.

Read-only by construction: we only issue `SELECT`. No DML, no DDL.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from biopsy.warehouse._base import (
    AdapterResult,
    ParsedURI,
    ScanOptions,
    WarehouseDriverNotInstalledError,
    resolve_credentials,
)

if TYPE_CHECKING:
    import duckdb


# Default cost-warning threshold. ~5 GB is a small Standard-edition query;
# above that the user almost certainly wants `--sample` or `--filter`.
DEFAULT_COST_WARN_BYTES = 5 * 1024 * 1024 * 1024


# BigQuery type → DuckDB-like type, only as much as `kind_of()` and
# `parse_filter_expr` need to know whether a column is numeric. ARRAY /
# STRUCT / GEOGRAPHY / JSON / BYTES intentionally fall through to "text"
# semantics — they don't fit biopsy's profiling primitives anyway.
_BQ_TO_DUCKDB_TYPE: dict[str, str] = {
    "INT64": "BIGINT",
    "INTEGER": "BIGINT",
    "FLOAT64": "DOUBLE",
    "FLOAT": "DOUBLE",
    "NUMERIC": "DECIMAL",
    "BIGNUMERIC": "DECIMAL",
    "BOOL": "BOOLEAN",
    "BOOLEAN": "BOOLEAN",
    "STRING": "VARCHAR",
    "DATE": "DATE",
    "DATETIME": "TIMESTAMP",
    "TIMESTAMP": "TIMESTAMP",
    "TIME": "TIME",
}


def open_bigquery(
    con: duckdb.DuckDBPyConnection,
    parsed: ParsedURI,
    *,
    options: ScanOptions | None = None,
) -> AdapterResult:
    """Execute a remote SELECT and return the result as an Arrow table."""
    options = options or ScanOptions()
    client = _connect(options.credentials_prefix)

    project, dataset, table = _split_table(parsed)
    fully_qualified = f"`{project}`.`{dataset}`.`{table}`"

    schema = _fetch_schema(client, project, dataset, table)
    where_sql = _build_where(options.where_sql, schema)
    sql = f"SELECT * FROM {fully_qualified}"
    if where_sql:
        sql += f" WHERE {where_sql}"
    if options.limit is not None:
        sql += f" LIMIT {int(options.limit)}"

    _maybe_warn_cost(client, sql)

    arrow_table = client.query(sql).to_arrow()

    return AdapterResult(
        qualified_name=parsed.qualified,
        arrow_table=arrow_table,
        pushed_down=True,
    )


def discover_schema(
    parsed: ParsedURI,
    *,
    credentials_env: str | None = None,
) -> tuple[dict[str, str], int | None]:
    """Cheap schema + row-count lookup via INFORMATION_SCHEMA + __TABLES__.

    Used by `biopsy doctor` against BigQuery URIs so a multi-TB table
    doesn't trigger a scan.
    """
    client = _connect(credentials_env)
    project, dataset, table = _split_table(parsed)
    bq_schema = _fetch_schema(client, project, dataset, table)

    # __TABLES__ is the legacy per-dataset metadata view; row_count there
    # is exact and free. INFORMATION_SCHEMA.TABLES is the standard SQL
    # equivalent but doesn't expose row count on every BigQuery variant.
    row_count: int | None = None
    try:
        meta_sql = (
            f"SELECT row_count FROM `{project}.{dataset}.__TABLES__` WHERE table_id = @table_name"
        )
        from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

        params = [ScalarQueryParameter("table_name", "STRING", table)]
        job = client.query(meta_sql, job_config=QueryJobConfig(query_parameters=params))
        rows = list(job.result(max_results=1))
        if rows:
            row_count = int(rows[0]["row_count"])
    except Exception:
        # __TABLES__ may be unavailable (e.g. for views, external tables,
        # or some regions). Schema discovery is the load-bearing part —
        # row count is a nice extra; don't fail doctor if it's missing.
        row_count = None

    # `bq_schema` already maps column → BigQuery type. The doctor view
    # benefits from a DuckDB-compatible type string so `kind_of()` returns
    # the right kind (numeric / temporal / text).
    duckdb_schema = {name: _BQ_TO_DUCKDB_TYPE.get(dt.upper(), dt) for name, dt in bq_schema.items()}
    return duckdb_schema, row_count


# --- internals -------------------------------------------------------------


def _connect(credentials_env: str | None) -> object:
    """Lazy-import google-cloud-bigquery and build a client.

    Credentials come from `GOOGLE_APPLICATION_CREDENTIALS` (a path to a
    service-account JSON file). With `--credentials-env STAGING`,
    `STAGING_GOOGLE_APPLICATION_CREDENTIALS` is read instead.
    """
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ImportError as exc:
        raise WarehouseDriverNotInstalledError(
            "BigQuery adapter requires google-cloud-bigquery. "
            "Install with: pip install 'biopsy[bigquery]'."
        ) from exc

    creds = resolve_credentials("bigquery", prefix=credentials_env)
    key_path = creds.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path or not os.path.isfile(key_path):
        raise WarehouseDriverNotInstalledError(
            f"GOOGLE_APPLICATION_CREDENTIALS={key_path!r} does not point to "
            "a readable service-account JSON file."
        )
    credentials = service_account.Credentials.from_service_account_file(key_path)
    project = creds.get("BIGQUERY_PROJECT") or credentials.project_id
    return bigquery.Client(project=project, credentials=credentials)


def _split_table(parsed: ParsedURI) -> tuple[str, str, str]:
    """Resolve `bigquery://project/dataset.table` to (project, dataset, table).

    Project may be in the URI host or come from `BIGQUERY_PROJECT` env;
    the URI host takes precedence. The path is `dataset.table`. The URI
    parser already validated both as plain identifiers.
    """
    project = parsed.host
    path = parsed.path.lstrip("/")
    if not path or "." not in path:
        raise ValueError(
            f"BigQuery URI must be bigquery://project/dataset.table (got {parsed.qualified})"
        )
    dataset, table = path.split(".", 1)
    if not project:
        # No host → project must come from env (handled by client.project).
        # Surface a clearer error than the client default.
        creds = resolve_credentials("bigquery")
        project = creds.get("BIGQUERY_PROJECT")
        if not project:
            raise ValueError(
                "BigQuery URI must specify a project in the host segment, or set $BIGQUERY_PROJECT."
            )
    return project, dataset, table


def _fetch_schema(
    client: object,
    project: str,
    dataset: str,
    table: str,
) -> dict[str, str]:
    """Return {column_name: bigquery_type} from INFORMATION_SCHEMA.COLUMNS.

    BigQuery types come back as canonical names (INT64, STRING, ...). The
    caller maps to biopsy/DuckDB types when needed.
    """
    sql = (
        f"SELECT column_name, data_type "
        f"FROM `{project}.{dataset}`.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE table_name = @table_name "
        f"ORDER BY ordinal_position"
    )
    from google.cloud.bigquery import QueryJobConfig, ScalarQueryParameter

    params = [ScalarQueryParameter("table_name", "STRING", table)]
    job = client.query(sql, job_config=QueryJobConfig(query_parameters=params))
    rows = list(job.result())
    if not rows:
        raise ValueError(
            f"Table `{project}.{dataset}.{table}` not found or has no "
            "columns. Check the URI and that the service account has "
            "BigQuery Data Viewer on the dataset."
        )
    return {row["column_name"]: row["data_type"] for row in rows}


def _build_where(
    expressions: list[str],
    bq_schema: dict[str, str],
) -> str:
    """Parse user `--filter` predicates against the BigQuery schema and
    re-quote identifiers with backticks.

    `parse_filter_expr` expects a DuckDB-style dtypes dict and emits
    double-quoted identifiers. We map BigQuery types to their DuckDB
    equivalents so its `is_numeric` check fires correctly, then swap
    `"name"` for `` `name` `` in the output.
    """
    if not expressions:
        return ""
    from biopsy.io import parse_filter_expr

    duckdb_dtypes = {name: _BQ_TO_DUCKDB_TYPE.get(dt.upper(), dt) for name, dt in bq_schema.items()}
    clauses = [parse_filter_expr(expr, duckdb_dtypes) for expr in expressions]
    backticked = [_swap_quotes_to_backticks(c) for c in clauses]
    return " AND ".join(f"({c})" for c in backticked)


def _swap_quotes_to_backticks(clause: str) -> str:
    """Translate DuckDB-style `"name"` identifiers to BigQuery `` `name` ``.

    `parse_filter_expr` only ever emits double-quoted identifiers at
    `_quote_ident()`. Literals containing `"` are wrapped in single
    quotes and escaped — they never contain a bare `"` after parsing.
    So a global swap is safe.
    """
    return clause.replace('"', "`")


def _maybe_warn_cost(client: object, sql: str) -> None:
    """Run a dry-run query and print a stderr warning above the threshold."""
    try:
        from google.cloud.bigquery import QueryJobConfig

        job = client.query(sql, job_config=QueryJobConfig(dry_run=True, use_query_cache=False))
        bytes_processed = getattr(job, "total_bytes_processed", None)
    except Exception:
        # Dry-run quoting can fail on certain SQL variants. Don't block
        # the real query just because the estimate isn't available.
        return
    if bytes_processed is None or bytes_processed < DEFAULT_COST_WARN_BYTES:
        return
    gb = bytes_processed / (1024**3)
    print(
        f"biopsy: BigQuery dry-run estimates {gb:.1f} GB scanned. "
        "Consider --sample N or --filter to narrow the scan.",
        file=sys.stderr,
    )
