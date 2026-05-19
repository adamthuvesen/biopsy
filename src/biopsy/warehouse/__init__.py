"""Read-only warehouse and object-store source adapters.

Biopsy is DuckDB-first: every per-column statistic and pairwise correlation
runs as SQL against an in-memory DuckDB view named `data`. Warehouse adapters
turn an external source (S3, Snowflake, Postgres, BigQuery) into either a
SQL scan expression DuckDB can read directly, or an Arrow table that DuckDB
can register zero-copy.

Read-only / pull-only by construction: adapters issue SELECTs only against
remote sources. The lint test `test_warehouse_readonly.py` enforces this
mechanically. The only DDL biopsy issues is against its **local** DuckDB
connection (the input view).
"""

from biopsy.warehouse._base import (
    SUPPORTED_SCHEMES,
    AdapterResult,
    MissingCredentialError,
    ParsedURI,
    ScanOptions,
    WarehouseDriverNotInstalledError,
    parse_warehouse_uri,
    resolve_credentials,
)

__all__ = [
    "AdapterResult",
    "MissingCredentialError",
    "ParsedURI",
    "SUPPORTED_SCHEMES",
    "ScanOptions",
    "WarehouseDriverNotInstalledError",
    "parse_warehouse_uri",
    "resolve_credentials",
]
