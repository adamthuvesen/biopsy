"""Unit tests for the warehouse package — URI parsing, credentials, dispatch.

Network-gated integration tests live next to these but are skipped unless
the matching `BIOPSY_TEST_<SCHEME>=1` env var is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from biopsy.warehouse import (
    SUPPORTED_SCHEMES,
    MissingCredentialError,
    parse_warehouse_uri,
    resolve_credentials,
)
from biopsy.warehouse.object_store import (
    _SECRET_SCOPE,
    open_object_store,
)

# --- URI parsing -----------------------------------------------------------


class TestURIParsing:
    def test_returns_none_for_non_uri(self) -> None:
        assert parse_warehouse_uri("data.parquet") is None
        assert parse_warehouse_uri("/abs/path.csv") is None
        assert parse_warehouse_uri("./relative.json") is None
        assert parse_warehouse_uri("") is None

    def test_every_supported_scheme_parses(self) -> None:
        cases = {
            "s3://bucket/key.parquet": ("s3", "bucket"),
            "s3a://bucket/key.parquet": ("s3a", "bucket"),
            "https://host/path.parquet": ("https", "host"),
            "http://host/path.parquet": ("http", "host"),
            "gs://bucket/key.parquet": ("gs", "bucket"),
            "gcs://bucket/key.parquet": ("gcs", "bucket"),
            "snowflake://acct/db.schema.table": ("snowflake", "acct"),
            "bigquery://proj/dataset.table": ("bigquery", "proj"),
            "postgres://host/db?table=public.events": ("postgres", "host"),
            "postgresql://host/db?table=public.events": ("postgresql", "host"),
        }
        for uri, (scheme, host) in cases.items():
            parsed = parse_warehouse_uri(uri)
            assert parsed is not None, uri
            assert parsed.scheme == scheme
            assert parsed.host == host

    def test_unsupported_scheme_raises_with_supported_list(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URI scheme 'redis'"):
            parse_warehouse_uri("redis://host/key")
        # Make sure the error lists the supported schemes for discoverability.
        with pytest.raises(ValueError, match="snowflake"):
            parse_warehouse_uri("ftp://host/key")

    def test_userinfo_is_stripped_from_qualified(self) -> None:
        parsed = parse_warehouse_uri("postgres://user:secret@host:5432/db?table=events")
        assert parsed is not None
        # No credential fragments anywhere in the qualified form.
        assert "user" not in parsed.qualified
        assert "secret" not in parsed.qualified
        # But the host/port/path/query are preserved.
        assert parsed.host == "host"
        assert "host:5432" in parsed.qualified
        assert "table=events" in parsed.qualified

    def test_table_identifier_validation(self) -> None:
        # Plain identifier is fine.
        ok = parse_warehouse_uri("postgres://host/db?table=public.events")
        assert ok is not None
        assert ok.table == "public.events"
        # SQL injection in the table param is rejected.
        with pytest.raises(ValueError, match="Invalid table identifier"):
            parse_warehouse_uri("postgres://host/db?table=public.events; DROP TABLE users")

    def test_snowflake_path_validation(self) -> None:
        # Three-part path is fine.
        ok = parse_warehouse_uri("snowflake://acct/db.schema.table")
        assert ok is not None
        assert ok.table == "db.schema.table"
        # Bad path identifier rejected.
        with pytest.raises(ValueError, match="Invalid path identifier"):
            parse_warehouse_uri("snowflake://acct/db; DROP DATABASE x")

    def test_query_keys_sort_in_qualified(self) -> None:
        # Stable ordering so saved profiles diff cleanly across runs.
        a = parse_warehouse_uri("postgres://host/db?table=t&zzz=1&aaa=2")
        b = parse_warehouse_uri("postgres://host/db?aaa=2&zzz=1&table=t")
        assert a is not None and b is not None
        assert a.qualified == b.qualified


# --- Credential resolution -------------------------------------------------


class TestCredentialResolution:
    def test_missing_required_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(MissingCredentialError, match="SNOWFLAKE_ACCOUNT"):
            resolve_credentials("snowflake")

    def test_optional_vars_omitted_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Optional-only scheme: missing vars are not errors.
        for var in (
            "PGHOST", "PGUSER", "PGPASSWORD", "PGDATABASE", "PGPORT", "PGSSLMODE",
        ):
            monkeypatch.delenv(var, raising=False)
        creds = resolve_credentials("postgres")
        assert creds == {}

    def test_prefix_overrides_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER",
            "STAGING_SNOWFLAKE_ACCOUNT", "STAGING_SNOWFLAKE_USER",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("STAGING_SNOWFLAKE_ACCOUNT", "staging-acct")
        monkeypatch.setenv("STAGING_SNOWFLAKE_USER", "staging-user")

        # No prefix → unset → raises.
        with pytest.raises(MissingCredentialError):
            resolve_credentials("snowflake")

        # With prefix → succeeds and returns unprefixed keys.
        creds = resolve_credentials("snowflake", prefix="STAGING")
        assert creds["SNOWFLAKE_ACCOUNT"] == "staging-acct"
        assert creds["SNOWFLAKE_USER"] == "staging-user"

    def test_error_message_mentions_prefix_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(MissingCredentialError) as exc:
            resolve_credentials("snowflake")
        assert "--credentials-env" in str(exc.value)


# --- Object-store adapter --------------------------------------------------


class TestObjectStoreAdapter:
    def test_scan_sql_for_parquet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No AWS creds → no SECRET install attempted, but the SCAN sql is built.
        for var in (
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION",
        ):
            monkeypatch.delenv(var, raising=False)
        parsed = parse_warehouse_uri("s3://bucket/path/data.parquet")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        result = open_object_store(con, parsed)
        assert result.scan_sql == "read_parquet('s3://bucket/path/data.parquet')"
        assert result.qualified_name == "s3://bucket/path/data.parquet"

    def test_scan_sql_for_csv_uses_sample_size_minus_one(self) -> None:
        parsed = parse_warehouse_uri("https://example.com/data.csv")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        result = open_object_store(con, parsed)
        assert "read_csv_auto" in (result.scan_sql or "")
        assert "sample_size=-1" in (result.scan_sql or "")

    def test_scan_sql_for_json(self) -> None:
        parsed = parse_warehouse_uri("https://example.com/data.json")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        result = open_object_store(con, parsed)
        assert "read_json_auto" in (result.scan_sql or "")

    def test_unknown_extension_rejected(self) -> None:
        parsed = parse_warehouse_uri("s3://bucket/data.avro")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        with pytest.raises(ValueError, match="must end in .parquet"):
            open_object_store(con, parsed)

    def test_gs_scheme_remapped_to_gcs_for_duckdb(self) -> None:
        # DuckDB's httpfs uses gcs:// internally; we accept gs:// from the
        # user but rewrite it before handing to read_parquet().
        parsed = parse_warehouse_uri("gs://bucket/data.parquet")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        result = open_object_store(con, parsed)
        assert "gcs://bucket/data.parquet" in (result.scan_sql or "")

    def test_s3_secret_installed_when_credentials_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretvalue")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        parsed = parse_warehouse_uri("s3://bucket/data.parquet")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        open_object_store(con, parsed)
        # DuckDB exposes secrets via duckdb_secrets().
        rows = con.execute(
            "SELECT name FROM duckdb_secrets() WHERE name = 'biopsy_s3'"
        ).fetchall()
        assert rows == [("biopsy_s3",)], (
            "expected biopsy_s3 secret to be installed when AWS creds are present"
        )

    def test_no_secret_when_credentials_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in (
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION",
        ):
            monkeypatch.delenv(var, raising=False)
        parsed = parse_warehouse_uri("s3://bucket/data.parquet")
        assert parsed is not None
        con = duckdb.connect(":memory:")
        open_object_store(con, parsed)
        rows = con.execute(
            "SELECT name FROM duckdb_secrets() WHERE name = 'biopsy_s3'"
        ).fetchall()
        assert rows == [], "no secret should be installed when AWS creds are unset"

    def test_supported_schemes_cover_object_store(self) -> None:
        for scheme in {"s3", "s3a", "https", "http", "gs", "gcs"}:
            assert scheme in SUPPORTED_SCHEMES
        # Sanity: secret-scope map covers the auth-required schemes.
        for scheme in {"s3", "s3a", "gs", "gcs"}:
            assert scheme in _SECRET_SCOPE


# --- Serialization redaction (defense in depth) ---------------------------


class TestSerializationRedaction:
    def test_userinfo_stripped_from_string_fields(self) -> None:
        from biopsy.serialize import to_jsonable

        # `to_jsonable` strips userinfo from any string that looks like a URI.
        v = to_jsonable("postgres://user:secret@host:5432/db?table=events")
        assert v == "postgres://host:5432/db?table=events"
        assert "secret" not in v
        assert "user" not in v

    def test_userinfo_with_only_user_stripped(self) -> None:
        from biopsy.serialize import to_jsonable

        assert to_jsonable("s3://AKIA@bucket/key.parquet") == "s3://bucket/key.parquet"

    def test_non_uri_strings_unchanged(self) -> None:
        from biopsy.serialize import to_jsonable

        # Plain strings — no '://' anywhere — pass through.
        assert to_jsonable("hello world") == "hello world"
        assert to_jsonable("/abs/path.parquet") == "/abs/path.parquet"

    def test_uri_without_userinfo_unchanged(self) -> None:
        from biopsy.serialize import to_jsonable

        assert (
            to_jsonable("https://example.com/data.parquet")
            == "https://example.com/data.parquet"
        )


# --- Network-gated end-to-end smoke ---------------------------------------


@pytest.mark.skipif(
    os.environ.get("BIOPSY_TEST_HTTPS") != "1",
    reason="set BIOPSY_TEST_HTTPS=1 to run network-dependent httpfs smoke",
)
def test_https_parquet_end_to_end() -> None:
    """Hits a public Parquet over HTTPS — slow, network-dependent.

    Gated behind BIOPSY_TEST_HTTPS=1 so the default suite stays offline.
    """
    from biopsy import profile

    url = os.environ.get(
        "BIOPSY_TEST_HTTPS_URL",
        # NYC TLC trip data is a stable public Parquet for smoke tests.
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    )
    prof = profile(url, sample=2000)
    assert prof.n_rows == 2000
    assert prof.source_uri == url
    assert prof.source_path is None


@pytest.mark.skipif(
    os.environ.get("BIOPSY_TEST_S3") != "1",
    reason="set BIOPSY_TEST_S3=1 and AWS creds to run S3 smoke",
)
def test_s3_parquet_end_to_end() -> None:
    """End-to-end against a real S3 bucket. Requires AWS creds + URL env var."""
    from biopsy import profile

    url = os.environ.get("BIOPSY_TEST_S3_URL")
    if not url:
        pytest.skip("set BIOPSY_TEST_S3_URL to an s3:// Parquet for this test")
    prof = profile(url, sample=2000)
    assert prof.n_rows <= 2000
    assert prof.source_uri == url
