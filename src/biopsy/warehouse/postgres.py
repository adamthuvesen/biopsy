"""Postgres adapter via DuckDB's built-in `postgres` extension.

The extension lets DuckDB ATTACH a remote Postgres database as a schema-
qualified namespace, so subsequent queries like
`SELECT * FROM pg_src.public.events` run as if Postgres were a local
DuckDB schema. No row data transfers until biopsy issues its SELECT.

Read-only: we ATTACH with `READ_ONLY` and additionally set
`default_transaction_read_only=on` on the underlying session. A misbehaved
adapter caller can't issue mutations through either channel.

URI form: `postgres://host[:port]/db?table=schema.name`
Connection details come from libpq env vars (PGHOST, PGUSER, PGPASSWORD,
PGPORT, PGDATABASE, PGSSLMODE) with the URI's host/port/db overriding
when present.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from biopsy.warehouse._base import (
    AdapterResult,
    ParsedURI,
    WarehouseDriverNotInstalledError,
    resolve_credentials,
)

if TYPE_CHECKING:
    import duckdb


# Module-level counter so multiple ATTACHes in one biopsy run don't collide
# on alias name. Each call gets a fresh `pg_src_<n>` alias.
_attach_counter = 0


def open_postgres(
    con: duckdb.DuckDBPyConnection,
    parsed: ParsedURI,
    *,
    credentials_env: str | None = None,
) -> AdapterResult:
    """ATTACH the Postgres database and return a scan expression.

    The returned `cleanup` callback DETACHes the alias — important when
    biopsy is used as a library and the same DuckDB connection might
    open multiple sources in sequence.
    """
    _ensure_postgres_extension(con)

    table = parsed.table
    if not table:
        raise ValueError(
            "Postgres URI must specify a table via ?table=schema.name "
            f"(got {parsed.qualified})"
        )

    creds = resolve_credentials("postgres", prefix=credentials_env)
    dsn = _build_dsn(parsed, creds)

    global _attach_counter
    _attach_counter += 1
    alias = f"pg_src_{_attach_counter}"

    # READ_ONLY at attach time blocks writes via DuckDB's wrapper; the
    # session-level transaction setting is a second line of defense for
    # connection pooling edge cases where the wrapper might re-issue
    # commands on a fresh connection.
    con.execute(f"ATTACH '{_sql_escape(dsn)}' AS {alias} (TYPE POSTGRES, READ_ONLY)")
    # postgres_execute is available in DuckDB 1.1+. If older, the READ_ONLY
    # attach is still in effect — degrade gracefully. The readonly lint
    # test catches code-level mutations regardless of this runtime check.
    with contextlib.suppress(Exception):
        con.execute(
            f"CALL postgres_execute("
            f"'{alias}', 'SET default_transaction_read_only = on'"
            f")"
        )

    qualified_table = f"{alias}.{table}"

    def cleanup() -> None:
        # Connection may already be closed when load() raises; safe to
        # ignore — the OS reclaims the socket either way.
        with contextlib.suppress(Exception):
            con.execute(f"DETACH {alias}")

    return AdapterResult(
        qualified_name=parsed.qualified,
        scan_sql=qualified_table,
        cleanup=cleanup,
    )


def _ensure_postgres_extension(con: duckdb.DuckDBPyConnection) -> None:
    """Install + load the `postgres` extension. Both ops are idempotent.

    DuckDB downloads the extension on first install; subsequent runs
    short-circuit. If the user has no network access, INSTALL fails with
    a network error — re-raise as our typed error so the CLI gives an
    actionable message.
    """
    try:
        con.execute("INSTALL postgres")
        con.execute("LOAD postgres")
    except Exception as exc:
        raise WarehouseDriverNotInstalledError(
            "DuckDB postgres extension failed to load. "
            "Check network access (extensions install on first use) and "
            "DuckDB version (1.1+ recommended)."
        ) from exc


def _build_dsn(parsed: ParsedURI, creds: dict[str, str]) -> str:
    """Construct a libpq-style keyword DSN.

    Keyword format (`host=foo dbname=bar ...`) avoids URL-escape pitfalls
    on passwords that contain special characters. URI host/port/dbname
    take precedence; env vars fill in everything else.
    """
    parts: dict[str, str] = {}
    if parsed.host:
        parts["host"] = parsed.host
    elif "PGHOST" in creds:
        parts["host"] = creds["PGHOST"]

    # urlparse `.port` is on the original URL; for the `parsed` dataclass
    # we encoded it into the qualified form. Re-extract from netloc.
    host_port = parsed.qualified.split("://", 1)[-1].split("/", 1)[0]
    if ":" in host_port:
        port = host_port.rsplit(":", 1)[-1]
        if port.isdigit():
            parts["port"] = port
    elif "PGPORT" in creds:
        parts["port"] = creds["PGPORT"]

    # URI path is `/dbname`. Strip the leading slash.
    db = parsed.path.lstrip("/")
    if db:
        parts["dbname"] = db
    elif "PGDATABASE" in creds:
        parts["dbname"] = creds["PGDATABASE"]

    if "PGUSER" in creds:
        parts["user"] = creds["PGUSER"]
    if "PGPASSWORD" in creds:
        parts["password"] = creds["PGPASSWORD"]
    if "PGSSLMODE" in creds:
        parts["sslmode"] = creds["PGSSLMODE"]

    if "dbname" not in parts:
        raise ValueError(
            "Postgres URI is missing a database name. Use "
            "`postgres://host/dbname?table=schema.name` or set $PGDATABASE."
        )

    return " ".join(f"{k}={_dsn_escape(v)}" for k, v in parts.items())


def _dsn_escape(value: str) -> str:
    """Quote a libpq DSN value if it contains whitespace or quotes."""
    if any(c in value for c in " \t'\\"):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    return value


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def discover_schema(
    con: duckdb.DuckDBPyConnection,
    parsed: ParsedURI,
    *,
    credentials_env: str | None = None,
) -> tuple[dict[str, str], int | None]:
    """Schema + row-count estimate via INFORMATION_SCHEMA + pg_class.

    Returns (column_name → dtype, estimated_rows). The row count is
    `pg_class.reltuples` — Postgres' planner-cached estimate, updated by
    ANALYZE. May be stale but is free (no scan).
    """
    result = open_postgres(con, parsed, credentials_env=credentials_env)
    if result.scan_sql is None:
        raise RuntimeError("Postgres adapter did not return scan_sql.")
    alias = result.scan_sql.split(".", 1)[0]
    table = parsed.table or ""
    if "." in table:
        schema_name, table_name = table.split(".", 1)
    else:
        schema_name, table_name = "public", table

    try:
        # DuckDB resolves `{alias}.information_schema.columns` against the
        # attached Postgres database, so this runs against Postgres'
        # catalog — not biopsy's local DuckDB catalog.
        cols = con.execute(
            f"SELECT column_name, data_type "
            f"FROM {alias}.information_schema.columns "
            f"WHERE table_schema = ? AND table_name = ? "
            f"ORDER BY ordinal_position",
            [schema_name, table_name],
        ).fetchall()
        if not cols:
            raise ValueError(
                f"Table {schema_name}.{table_name} not found or has no columns. "
                "Check the URI and that the user has SELECT on the table."
            )
        schema = {name: dtype for name, dtype in cols}

        # reltuples is the planner's row-count estimate. The `regclass`
        # cast is Postgres-only and DuckDB's parser rejects it, so we
        # send the query as raw SQL through `postgres_query()`. Quote
        # the identifier carefully — `_require_ident` already validated
        # both halves as plain SQL identifiers.
        qualified_table = f"{schema_name}.{table_name}"
        pg_sql = (
            f"SELECT reltuples::BIGINT AS est FROM pg_class "
            f"WHERE oid = '{qualified_table}'::regclass"
        )
        try:
            row = con.execute(
                f"SELECT est FROM postgres_query('{alias}', '{_sql_escape(pg_sql)}')"
            ).fetchone()
            est = int(row[0]) if row and row[0] is not None else None
        except Exception:
            # pg_class layout varies across Postgres major versions and
            # some hosted variants block `regclass` casts. Schema discovery
            # is the load-bearing part; the row count is a nice extra.
            # Don't fail doctor if only the estimate is missing.
            est = None
    finally:
        if result.cleanup is not None:
            result.cleanup()

    return schema, est
