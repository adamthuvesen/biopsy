"""Object-store adapter (S3, HTTPS, GCS) using DuckDB's httpfs extension.

`httpfs` is built into DuckDB; we install + load it lazily on first use.
Credentials come from environment variables via `resolve_credentials`,
materialized into a DuckDB `CREATE SECRET` statement for the duration of
this connection only.

Read-only: this adapter only issues `read_parquet` / `read_csv_auto` /
`read_json_auto`. It never writes back to the object store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from biopsy.warehouse._base import (
    AdapterResult,
    ParsedURI,
    resolve_credentials,
)

if TYPE_CHECKING:
    import duckdb


# Map of URI scheme → DuckDB scope-prefix for `CREATE SECRET` statements.
# httpfs uses these prefixes to scope which requests pick up which secret.
_SECRET_SCOPE: dict[str, str] = {
    "s3": "s3://",
    "s3a": "s3://",
    "gs": "gcs://",
    "gcs": "gcs://",
}


def open_object_store(
    con: duckdb.DuckDBPyConnection,
    parsed: ParsedURI,
    *,
    credentials_env: str | None = None,
) -> AdapterResult:
    """Configure httpfs and return a scan expression for `parsed`."""
    _ensure_httpfs(con)
    _install_credentials(con, parsed, credentials_env=credentials_env)

    # DuckDB's httpfs accepts s3://, https://, http://, and gs://
    # in the read_* functions directly. Map gs:// → gcs:// since DuckDB
    # uses the latter as the canonical GCS scheme.
    url = parsed.qualified
    if parsed.scheme == "gs":
        url = "gcs://" + url[len("gs://"):]

    suffix = url.rsplit(".", 1)[-1].lower()
    if suffix == "parquet":
        scan_sql = f"read_parquet('{_sql_escape(url)}')"
    elif suffix in {"csv", "tsv", "txt"}:
        # sample_size=-1 matches the local-file path so type inference is
        # the same across local and remote sources.
        scan_sql = f"read_csv_auto('{_sql_escape(url)}', sample_size=-1)"
    elif suffix == "json":
        scan_sql = f"read_json_auto('{_sql_escape(url)}')"
    else:
        raise ValueError(
            f"Object-store URL must end in .parquet, .csv, .tsv, or .json; "
            f"got '{url}'."
        )

    return AdapterResult(qualified_name=parsed.qualified, scan_sql=scan_sql)


def _ensure_httpfs(con: duckdb.DuckDBPyConnection) -> None:
    """Install + load httpfs once per connection.

    Both operations are idempotent in DuckDB; calling them twice is safe.
    Failures here surface as DuckDB exceptions — let them propagate; the
    user needs to see the network/permission error directly.
    """
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")


def _install_credentials(
    con: duckdb.DuckDBPyConnection,
    parsed: ParsedURI,
    *,
    credentials_env: str | None = None,
) -> None:
    """Translate env-var credentials into a DuckDB `CREATE SECRET`.

    Public buckets and unauthenticated HTTPS sources need no secret;
    `resolve_credentials` returns an empty dict in that case and we
    short-circuit.
    """
    if parsed.scheme in {"http", "https"}:
        # HTTPS bearer auth is not yet pushed into DuckDB (the extension
        # supports it via secrets but the API is unstable across versions).
        # Skip for now — only public HTTPS Parquet/CSV works without auth.
        return

    creds = resolve_credentials(parsed.scheme, prefix=credentials_env)
    scope = _SECRET_SCOPE.get(parsed.scheme)
    if not creds or scope is None:
        return

    if parsed.scheme in {"s3", "s3a"}:
        key = creds.get("AWS_ACCESS_KEY_ID")
        secret = creds.get("AWS_SECRET_ACCESS_KEY")
        token = creds.get("AWS_SESSION_TOKEN")
        region = creds.get("AWS_REGION") or creds.get("AWS_DEFAULT_REGION")
        if not key or not secret:
            # Boto-style: fall back to anonymous; user may be hitting a
            # public bucket. DuckDB will return 403 on the read if not.
            return
        parts = [
            "TYPE S3",
            f"KEY_ID '{_sql_escape(key)}'",
            f"SECRET '{_sql_escape(secret)}'",
        ]
        if token:
            parts.append(f"SESSION_TOKEN '{_sql_escape(token)}'")
        if region:
            parts.append(f"REGION '{_sql_escape(region)}'")
        parts.append(f"SCOPE '{scope}'")
        body = ", ".join(parts)
        # Per-connection secret; DuckDB clears it when the connection closes.
        con.execute(f"CREATE OR REPLACE SECRET biopsy_s3 ({body})")
        return

    if parsed.scheme in {"gs", "gcs"}:
        # DuckDB reads Google Application Default Credentials from
        # GOOGLE_APPLICATION_CREDENTIALS automatically when the env var is
        # set in this process. Nothing else to do here.
        return


def _sql_escape(value: str) -> str:
    """Escape a single-quoted SQL literal — only quote-doubling needed."""
    return value.replace("'", "''")


def discover_schema(
    con: duckdb.DuckDBPyConnection, parsed: ParsedURI
) -> dict[str, str]:
    """Cheap schema discovery: read the Parquet footer or first CSV chunk.

    Used by `biopsy doctor` against object-store URIs so a 50 GB Parquet
    file doesn't get scanned end-to-end.
    """
    result = open_object_store(con, parsed)
    if result.scan_sql is None:
        raise RuntimeError("Object-store adapter did not return a scan_sql.")
    rows = con.execute(f"DESCRIBE SELECT * FROM {result.scan_sql} LIMIT 0").fetchall()
    return {r[0]: r[1] for r in rows}
