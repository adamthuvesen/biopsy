"""Shared types and helpers for warehouse adapters.

URI parsing, credential resolution, and the adapter return contract live
here. Individual adapters import from this module and contribute scheme-
specific scan logic on top.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, unquote, urlparse

if TYPE_CHECKING:
    import pyarrow as pa


SUPPORTED_SCHEMES: frozenset[str] = frozenset(
    {
        "snowflake",
        "bigquery",
        "postgres",
        "postgresql",
        "s3",
        "s3a",
        "https",
        "http",
        "gs",
        "gcs",
    }
)

# A SQL identifier (optionally schema-qualified): `events` or `public.events`.
# Three-part Snowflake/BigQuery names use dots too: `db.schema.table`.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


class WarehouseDriverNotInstalledError(ImportError):
    """Raised when a backend's optional dependency is not installed.

    The message is the exact one-line error shown to the user; the CLI
    converts this to a non-zero exit without a traceback.
    """


class MissingCredentialError(ValueError):
    """Raised when a required env var for a warehouse scheme is unset."""


@dataclass(frozen=True)
class ParsedURI:
    """A warehouse URI broken into its addressable parts.

    `qualified` is the credential-free string used for `Source.source_uri`,
    progress logs, and the saved profile JSON. It NEVER contains userinfo,
    even if the original URI did.
    """

    scheme: str
    host: str | None
    path: str
    query: dict[str, str]
    qualified: str

    @property
    def table(self) -> str | None:
        """Best-effort table identifier for SQL warehouses.

        For Postgres-style URIs the table comes from `?table=schema.name`;
        for Snowflake/BigQuery it's derived from the URI path. Returns
        None for object-store URIs (which address files, not tables).
        """
        t = self.query.get("table")
        if t is not None:
            return t
        if self.scheme in {"snowflake", "bigquery"}:
            # path is "/db.schema.table" → "db.schema.table"
            return self.path.lstrip("/") or None
        return None


@dataclass(frozen=True)
class ScanOptions:
    """Push-down options the caller wants the adapter to honor.

    For warehouse sources both `where_sql` and `limit` are load-bearing —
    without them the adapter would transfer the full table. Adapters MUST
    apply both at the remote side, not after pulling rows locally.
    """

    where_sql: list[str] = field(default_factory=list)
    limit: int | None = None
    credentials_prefix: str | None = None


@dataclass
class AdapterResult:
    """What an adapter returns to `io.load()`.

    Exactly one of `scan_sql` or `arrow_table` is populated:
      - `scan_sql`: a `FROM <expr>` source DuckDB can read directly
        (DuckDB extensions: postgres, httpfs).
      - `arrow_table`: an in-memory Arrow table biopsy registers with
        DuckDB the same way it handles user-supplied frames (Snowflake,
        BigQuery via the Python client).

    `qualified_name` is the credential-free URI; biopsy stores it on
    `Source.source_uri` and uses it as the display name.

    `cleanup` is called after the local `data` table is materialized —
    e.g. detaching a Postgres connection. It MUST be safe to call multiple
    times and SHOULD NOT raise.

    `pushed_down=True` signals that the adapter already applied the
    caller's `ScanOptions.where_sql` and `ScanOptions.limit` at the remote
    source. `load()` then skips its outer `WHERE`/`USING SAMPLE` wrapper
    so we don't filter twice (which would be wasteful and would re-evaluate
    user predicates against DuckDB's parser, breaking on vendor-specific
    SQL). Default `False` matches existing scan-SQL adapters (Postgres,
    object-store) where DuckDB does the push-down via its wrapper.
    """

    qualified_name: str
    scan_sql: str | None = None
    arrow_table: pa.Table | None = None
    cleanup: Callable[[], None] | None = None
    pushed_down: bool = False


def parse_warehouse_uri(value: str) -> ParsedURI | None:
    """Parse a string as a warehouse URI, or return None if it isn't one.

    A string is treated as a URI iff it has a scheme in `SUPPORTED_SCHEMES`
    or a scheme we recognize as unsupported (in which case we raise so the
    user gets a clear error instead of a confusing "file not found").

    Strings without a `://` separator are not URIs — return None so the
    caller falls back to path/in-memory handling.
    """
    if "://" not in value:
        return None
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if not scheme:
        return None
    if scheme not in SUPPORTED_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_SCHEMES))
        raise ValueError(f"Unsupported URI scheme '{scheme}'. Supported: {supported}.")

    # Strip userinfo: rebuild netloc without `user:pass@`.
    host = parsed.hostname
    netloc = host or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    query_pairs: dict[str, str] = {}
    if parsed.query:
        for key, vals in parse_qs(parsed.query, keep_blank_values=False).items():
            # Take the last value for repeated keys — matches urllib's default.
            query_pairs[key] = vals[-1]

    # Validate table identifier — refuses anything that isn't a plain SQL
    # identifier so a malicious `?table=` value can't smuggle SQL.
    if "table" in query_pairs:
        _require_ident(query_pairs["table"], "table")

    # Snowflake / BigQuery use the URI path as `db.schema.table`. Validate
    # it as an identifier too so a malicious URI can't smuggle SQL.
    if scheme in {"snowflake", "bigquery"}:
        path_ident = parsed.path.lstrip("/")
        if path_ident:
            _require_ident(path_ident, "path")

    qualified_path = parsed.path or ""
    # Rebuild a credential-free URI for display + storage.
    qualified = f"{scheme}://{netloc}{qualified_path}"
    if query_pairs:
        # Re-encode in stable key order so saved profiles diff cleanly.
        qs = "&".join(f"{k}={query_pairs[k]}" for k in sorted(query_pairs))
        qualified = f"{qualified}?{qs}"

    return ParsedURI(
        scheme=scheme,
        host=host,
        path=unquote(parsed.path or ""),
        query=query_pairs,
        qualified=qualified,
    )


def _require_ident(value: str, label: str) -> None:
    if not _IDENT_RE.match(value):
        raise ValueError(
            f"Invalid {label} identifier in URI: {value!r}. "
            "Must match [A-Za-z_][A-Za-z0-9_]* with up to two dot-separated parts."
        )


# --- credential resolution -------------------------------------------------


# Maps scheme → ordered list of env-var names that the adapter wants. The
# first entry in each tuple is REQUIRED; the remainder are optional and
# returned only if set. This is the single source of truth — adapters
# import from here so a doc audit walks one map, not five.
_CREDENTIAL_KEYS: dict[str, list[tuple[str, bool]]] = {
    "snowflake": [
        ("SNOWFLAKE_ACCOUNT", True),
        ("SNOWFLAKE_USER", True),
        # One of PRIVATE_KEY_PATH or PASSWORD is required; the adapter
        # checks that at use time so error messages can be specific.
        ("SNOWFLAKE_PRIVATE_KEY_PATH", False),
        ("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", False),
        ("SNOWFLAKE_PASSWORD", False),
        ("SNOWFLAKE_WAREHOUSE", False),
        ("SNOWFLAKE_ROLE", False),
        ("SNOWFLAKE_DATABASE", False),
        ("SNOWFLAKE_SCHEMA", False),
    ],
    "bigquery": [
        ("GOOGLE_APPLICATION_CREDENTIALS", True),
        ("BIGQUERY_PROJECT", False),
    ],
    "postgres": [
        ("PGHOST", False),
        ("PGUSER", False),
        ("PGPASSWORD", False),
        ("PGDATABASE", False),
        ("PGPORT", False),
        ("PGSSLMODE", False),
    ],
    "postgresql": [
        ("PGHOST", False),
        ("PGUSER", False),
        ("PGPASSWORD", False),
        ("PGDATABASE", False),
        ("PGPORT", False),
        ("PGSSLMODE", False),
    ],
    "s3": [
        ("AWS_ACCESS_KEY_ID", False),
        ("AWS_SECRET_ACCESS_KEY", False),
        ("AWS_SESSION_TOKEN", False),
        ("AWS_REGION", False),
        ("AWS_DEFAULT_REGION", False),
    ],
    "s3a": [
        ("AWS_ACCESS_KEY_ID", False),
        ("AWS_SECRET_ACCESS_KEY", False),
        ("AWS_SESSION_TOKEN", False),
        ("AWS_REGION", False),
        ("AWS_DEFAULT_REGION", False),
    ],
    "gs": [
        ("GOOGLE_APPLICATION_CREDENTIALS", False),
    ],
    "gcs": [
        ("GOOGLE_APPLICATION_CREDENTIALS", False),
    ],
    "https": [
        ("BIOPSY_HTTPS_BEARER", False),
    ],
    "http": [
        ("BIOPSY_HTTPS_BEARER", False),
    ],
}


def resolve_credentials(scheme: str, prefix: str | None = None) -> dict[str, str]:
    """Look up credentials for a scheme from environment variables.

    With `prefix='STAGING'`, looks up `STAGING_SNOWFLAKE_USER` etc. instead
    of `SNOWFLAKE_USER`. Required keys missing from the environment raise
    `MissingCredentialError` with the variable name and override hint.

    Returns only the keys that are set (plus required ones, which raise if
    not set). The returned dict is keyed by the **unprefixed** name so
    adapters don't need to know about the prefix scheme.
    """
    keys = _CREDENTIAL_KEYS.get(scheme)
    if keys is None:
        raise ValueError(f"No credential schema for scheme '{scheme}'.")

    def env_name(base: str) -> str:
        return f"{prefix}_{base}" if prefix else base

    out: dict[str, str] = {}
    for base, required in keys:
        value = os.environ.get(env_name(base))
        if value is None or value == "":
            if required:
                hint = (
                    " (override prefix via --credentials-env PREFIX)"
                    if prefix is None
                    else f" (currently using --credentials-env {prefix})"
                )
                raise MissingCredentialError(
                    f"Missing required env var {env_name(base)} for {scheme}://...{hint}"
                )
            continue
        out[base] = value
    return out
