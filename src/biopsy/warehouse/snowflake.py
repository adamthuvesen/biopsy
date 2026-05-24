"""Snowflake adapter via snowflake-connector-python → Arrow → register.

No production-grade DuckDB Snowflake extension exists, so we use the
official Python connector to execute a remote `SELECT` and call
`cursor.fetch_arrow_all()`. The resulting Arrow table is registered with
biopsy's local DuckDB.

Push-down is mandatory: the adapter applies `--filter` (parsed by biopsy's
`parse_filter_expr`) and `--sample N` (as `LIMIT N`) to the remote
`SELECT`. Snowflake uses double-quoted identifiers (same as DuckDB) so no
quote translation is needed.

Snowflake's `INFORMATION_SCHEMA.TABLES.ROW_COUNT` is exact for tables (NULL
for views). When available, biopsy uses it to warn about Arrow materializ-
ation cost — pulling 10⁸ cells over the wire is RAM-bound and slow.

Read-only by construction: we only issue `SELECT`. No DML, no DDL.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from biopsy.warehouse._base import (
    AdapterResult,
    ParsedURI,
    ScanOptions,
    WarehouseDriverNotInstalledError,
    resolve_credentials,
)

if TYPE_CHECKING:
    import duckdb


# Default RAM-bound warning. ~5×10⁸ cells (rows × columns) is the point
# at which pyarrow materialization gets sluggish and likely OOMs on a
# laptop. Cell count beats raw row count because wide tables compound.
DEFAULT_CELL_BUDGET = 5 * 10**8


# Snowflake type → DuckDB-like type. Snowflake's `data_type` column in
# INFORMATION_SCHEMA returns canonical names like NUMBER, TEXT, TIMESTAMP_NTZ.
_SF_TO_DUCKDB_TYPE: dict[str, str] = {
    "NUMBER": "DECIMAL",
    "DECIMAL": "DECIMAL",
    "NUMERIC": "DECIMAL",
    "INT": "BIGINT",
    "INTEGER": "BIGINT",
    "BIGINT": "BIGINT",
    "SMALLINT": "SMALLINT",
    "TINYINT": "TINYINT",
    "BYTEINT": "TINYINT",
    "FLOAT": "DOUBLE",
    "FLOAT4": "FLOAT",
    "FLOAT8": "DOUBLE",
    "DOUBLE": "DOUBLE",
    "REAL": "DOUBLE",
    "BOOLEAN": "BOOLEAN",
    "TEXT": "VARCHAR",
    "VARCHAR": "VARCHAR",
    "STRING": "VARCHAR",
    "CHAR": "VARCHAR",
    "CHARACTER": "VARCHAR",
    "DATE": "DATE",
    "TIME": "TIME",
    "DATETIME": "TIMESTAMP",
    "TIMESTAMP": "TIMESTAMP",
    "TIMESTAMP_NTZ": "TIMESTAMP",
    "TIMESTAMP_LTZ": "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP_TZ": "TIMESTAMP WITH TIME ZONE",
}


def open_snowflake(
    con: duckdb.DuckDBPyConnection,
    parsed: ParsedURI,
    *,
    options: ScanOptions | None = None,
) -> AdapterResult:
    """Execute a remote SELECT and return the result as an Arrow table."""
    options = options or ScanOptions()
    sf_con, cleanup = _connect(options.credentials_prefix)
    try:
        database, schema_name, table = _split_table(parsed)
        fully_qualified = f'"{database}"."{schema_name}"."{table}"'

        sf_schema = _fetch_schema(sf_con, database, schema_name, table)
        _maybe_warn_cell_budget(sf_con, database, schema_name, table, sf_schema, options.limit)

        where_sql = _build_where(options.where_sql, sf_schema)
        sql = f"SELECT * FROM {fully_qualified}"
        if where_sql:
            sql += f" WHERE {where_sql}"
        if options.limit is not None:
            sql += f" LIMIT {int(options.limit)}"

        cur = sf_con.cursor()
        try:
            cur.execute(sql)
            arrow_table = cur.fetch_arrow_all()
        finally:
            cur.close()
    finally:
        cleanup()

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
    """Schema + row-count via INFORMATION_SCHEMA.

    Snowflake's INFORMATION_SCHEMA.TABLES.ROW_COUNT is exact for tables
    (NULL for views and external tables). Free — no warehouse credits
    consumed for INFORMATION_SCHEMA reads.
    """
    sf_con, cleanup = _connect(credentials_env)
    try:
        database, schema_name, table = _split_table(parsed)
        sf_schema = _fetch_schema(sf_con, database, schema_name, table)
        row_count = _fetch_row_count(sf_con, database, schema_name, table)
    finally:
        cleanup()

    return (
        {name: _SF_TO_DUCKDB_TYPE.get(dt.upper(), dt) for name, dt in sf_schema.items()},
        row_count,
    )


# --- internals -------------------------------------------------------------


def _connect(credentials_env: str | None) -> tuple[Any, Any]:
    """Lazy-import snowflake.connector and open a connection.

    Returns (connection, cleanup_callable). Cleanup closes the connection
    — important when called repeatedly in tests or library mode.
    """
    try:
        import snowflake.connector
    except ImportError as exc:
        raise WarehouseDriverNotInstalledError(
            "Snowflake adapter requires snowflake-connector-python. "
            "Install with: pip install 'biopsy[snowflake]'."
        ) from exc

    creds = resolve_credentials("snowflake", prefix=credentials_env)
    kwargs: dict[str, Any] = {
        "account": creds["SNOWFLAKE_ACCOUNT"],
        "user": creds["SNOWFLAKE_USER"],
    }
    private_key_path = creds.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    password = creds.get("SNOWFLAKE_PASSWORD")
    if private_key_path:
        kwargs["private_key"] = _load_private_key(
            private_key_path,
            creds.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"),
        )
    elif password:
        kwargs["password"] = password
    else:
        raise WarehouseDriverNotInstalledError(
            "Snowflake requires either SNOWFLAKE_PRIVATE_KEY_PATH or "
            "SNOWFLAKE_PASSWORD to authenticate."
        )

    for env_key, conn_key in (
        ("SNOWFLAKE_WAREHOUSE", "warehouse"),
        ("SNOWFLAKE_ROLE", "role"),
        ("SNOWFLAKE_DATABASE", "database"),
        ("SNOWFLAKE_SCHEMA", "schema"),
    ):
        if creds.get(env_key):
            kwargs[conn_key] = creds[env_key]

    sf_con = snowflake.connector.connect(**kwargs)

    def cleanup() -> None:
        import contextlib

        with contextlib.suppress(Exception):
            sf_con.close()

    return sf_con, cleanup


def _load_private_key(path: str, passphrase: str | None) -> bytes:
    """Load a PEM-encoded private key for Snowflake key-pair auth."""
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:
        raise WarehouseDriverNotInstalledError(
            "Snowflake key-pair auth requires cryptography. "
            "Install with: pip install 'biopsy[snowflake]' (it pulls cryptography "
            "via snowflake-connector-python's requirements)."
        ) from exc

    with open(path, "rb") as f:
        pem = f.read()
    pk = serialization.load_pem_private_key(
        pem,
        password=passphrase.encode("utf-8") if passphrase else None,
        backend=default_backend(),
    )
    return pk.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _split_table(parsed: ParsedURI) -> tuple[str, str, str]:
    """Resolve `snowflake://account/db.schema.table` to (db, schema, table).

    The URI host is the account; the path is `db.schema.table`. The URI
    parser already validated the path as a three-part identifier.
    """
    path = parsed.path.lstrip("/")
    parts = path.split(".") if path else []
    if len(parts) != 3:
        raise ValueError(
            f"Snowflake URI must be snowflake://account/db.schema.table (got {parsed.qualified})"
        )
    return parts[0], parts[1], parts[2]


def _fetch_schema(
    sf_con: Any,
    database: str,
    schema_name: str,
    table: str,
) -> dict[str, str]:
    """Return {column_name: snowflake_type} from INFORMATION_SCHEMA.COLUMNS."""
    sql = (
        f"SELECT column_name, data_type "
        f'FROM "{database}"."INFORMATION_SCHEMA"."COLUMNS" '
        f"WHERE table_schema = %s AND table_name = %s "
        f"ORDER BY ordinal_position"
    )
    cur = sf_con.cursor()
    try:
        cur.execute(sql, (schema_name, table))
        rows = cur.fetchall()
    finally:
        cur.close()
    if not rows:
        raise ValueError(
            f"Table {database}.{schema_name}.{table} not found or has no "
            "columns. Check the URI and that the role has SELECT on the table."
        )
    return {name: dtype for name, dtype in rows}


def _fetch_row_count(
    sf_con: Any,
    database: str,
    schema_name: str,
    table: str,
) -> int | None:
    """Return INFORMATION_SCHEMA.TABLES.ROW_COUNT — None for views."""
    sql = (
        f"SELECT row_count "
        f'FROM "{database}"."INFORMATION_SCHEMA"."TABLES" '
        f"WHERE table_schema = %s AND table_name = %s"
    )
    cur = sf_con.cursor()
    try:
        cur.execute(sql, (schema_name, table))
        row = cur.fetchone()
    finally:
        cur.close()
    if not row or row[0] is None:
        return None
    return int(row[0])


def _build_where(
    expressions: list[str],
    sf_schema: dict[str, str],
) -> str:
    """Parse user `--filter` predicates against the Snowflake schema.

    Snowflake uses double-quoted identifiers (same as DuckDB) so the
    output of `parse_filter_expr` is sent as-is.
    """
    if not expressions:
        return ""
    from biopsy.io import parse_filter_expr

    duckdb_dtypes = {name: _SF_TO_DUCKDB_TYPE.get(dt.upper(), dt) for name, dt in sf_schema.items()}
    clauses = [parse_filter_expr(expr, duckdb_dtypes) for expr in expressions]
    return " AND ".join(f"({c})" for c in clauses)


def _maybe_warn_cell_budget(
    sf_con: Any,
    database: str,
    schema_name: str,
    table: str,
    sf_schema: dict[str, str],
    limit: int | None,
) -> None:
    """Print a stderr warning if the materialized Arrow table would be huge."""
    n_cols = len(sf_schema)
    if limit is not None:
        cells = limit * n_cols
    else:
        row_count = _fetch_row_count(sf_con, database, schema_name, table)
        if row_count is None:
            return
        cells = row_count * n_cols
    if cells < DEFAULT_CELL_BUDGET:
        return
    print(
        f"biopsy: Snowflake fetch is ~{cells / 1e9:.1f}B cells "
        f"({n_cols} cols × {cells // n_cols:,} rows). "
        "Pulling this much Arrow may be RAM-bound; consider --sample N "
        "or --filter to narrow.",
        file=sys.stderr,
    )
