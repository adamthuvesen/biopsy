"""DuckDB-backed loading for files and in-memory tabular data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

REGISTERED_INPUT_VIEW = "__biopsy_input__"


@dataclass(frozen=True)
class Source:
    """A loaded dataset, exposed as a DuckDB view named `data`.

    `source_path` is set for file-system inputs; `source_uri` is set for
    warehouse / object-store inputs (e.g. `s3://bucket/key.parquet`). At
    most one is set; in-memory dataframe inputs leave both as None.
    """

    con: duckdb.DuckDBPyConnection
    source_name: str
    source_path: Path | None
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    source_uri: str | None = None

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        return self.con.sql(query)


def _scan_expr(path: Path) -> str:
    suffix = path.suffix.lower()
    p = str(path).replace("'", "''")
    if suffix == ".parquet":
        return f"read_parquet('{p}')"
    if suffix in {".csv", ".tsv", ".txt"}:
        # auto-detect delimiter, header, types
        return f"read_csv_auto('{p}', sample_size=-1)"
    if suffix == ".json":
        return f"read_json_auto('{p}')"
    raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .tsv, .parquet, or .json")


def load(
    data: str | Path | Any,
    sample: int | None = None,
    exclude: list[str] | None = None,
    ignore_missing_exclude: bool = False,
    where: list[str] | None = None,
    source_name: str | None = None,
    credentials_env: str | None = None,
) -> Source:
    """Open a file or in-memory table as a DuckDB view named `data`.

    Parameters
    ----------
    data: path, pandas/polars/Arrow object, or DuckDB relation.
    sample: random reservoir sample size (deterministic seed=42).
    exclude: column names to drop from the view (e.g. known target proxies).
    ignore_missing_exclude: silently skip excluded columns absent from this dataset.
    where: filter expressions, AND-ed together. See `parse_filter_expr`.
    source_name: display name for in-memory data. File paths ignore this.
    """
    con = duckdb.connect(":memory:")
    cleanup: Any | None = None
    try:
        scan, resolved_path, display_name, resolved_uri, pushed_down, cleanup = _input_scan(
            con,
            data,
            source_name,
            credentials_env=credentials_env,
            where=where,
            sample=sample,
        )

        # First materialize the raw scan so we know the schema for filter parsing
        # and exclusion validation.
        raw_info = con.execute(f"DESCRIBE SELECT * FROM {scan} LIMIT 0").fetchall()
        raw_columns = [r[0] for r in raw_info]
        raw_dtypes = {r[0]: r[1] for r in raw_info}

        select_clause = "*"
        if exclude:
            missing = [c for c in exclude if c not in raw_columns]
            if missing:
                if ignore_missing_exclude:
                    exclude = [c for c in exclude if c in raw_columns]
                else:
                    present = [c for c in exclude if c in raw_columns]
                    shown = raw_columns[:8]
                    suffix = "..." if len(raw_columns) > len(shown) else ""
                    raise ValueError(
                        f"--exclude column(s) not in dataset: {missing}. "
                        f"Valid exclusions from this request: {present}. "
                        f"Available: {shown}{suffix}"
                    )
            if exclude:
                excl = ", ".join(_quote_ident(c) for c in exclude)
                select_clause = f"* EXCLUDE ({excl})"

        # Arrow-backed warehouse adapters apply WHERE / LIMIT at the remote
        # source. Skip biopsy's outer wrapper in that case so we don't
        # double-apply (and so user predicates aren't re-parsed against
        # DuckDB's SQL dialect, which would fail on vendor-specific syntax).
        where_clause = ""
        if where and not pushed_down:
            parts = [parse_filter_expr(expr, raw_dtypes) for expr in where]
            where_clause = " WHERE " + " AND ".join(f"({p})" for p in parts)

        is_file = scan != REGISTERED_INPUT_VIEW

        if sample and not pushed_down:
            # USING SAMPLE applies to the source *before* WHERE in the same SELECT,
            # so a filtered sample collapses to ~sample × fraction rows. Wrap the
            # filtered SELECT in a subquery so sampling runs on the filtered data.
            ddl = "CREATE TABLE" if is_file else "CREATE VIEW"
            con.execute(
                f"{ddl} data AS "
                f"SELECT * FROM (SELECT {select_clause} FROM {scan}{where_clause}) "
                f"USING SAMPLE {sample} ROWS (reservoir, 42)"
            )
        else:
            ddl = "CREATE TABLE" if is_file else "CREATE VIEW"
            con.execute(f"{ddl} data AS SELECT {select_clause} FROM {scan}{where_clause}")

        if cleanup is not None:
            cleanup()
            cleanup = None

        # Derive the final schema from raw_dtypes minus excluded columns
        # rather than running a second DESCRIBE pass (which can force DuckDB
        # to re-open the file on the CSV/Parquet path).
        excluded = set(exclude or [])
        columns = [c for c in raw_columns if c not in excluded]
        dtypes = {c: raw_dtypes[c] for c in columns}
        n_rows = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]

        return Source(
            con=con,
            source_name=display_name,
            source_path=resolved_path,
            n_rows=n_rows,
            n_cols=len(columns),
            columns=columns,
            dtypes=dtypes,
            source_uri=resolved_uri,
        )
    except BaseException:
        if cleanup is not None:
            cleanup()
        con.close()
        raise


def _input_scan(
    con: duckdb.DuckDBPyConnection,
    data: str | Path | Any,
    source_name: str | None,
    *,
    credentials_env: str | None = None,
    where: list[str] | None = None,
    sample: int | None = None,
) -> tuple[str, Path | None, str, str | None, bool, Any | None]:
    """Resolve `data` into a (scan_expr, path, display_name, uri, pushed_down)
    tuple.

    Dispatches in order: URI scheme → file path → DuckDB relation →
    DuckDB-registerable object. Only one branch ever returns a non-None
    `path` OR `uri` — they're mutually exclusive.

    `pushed_down=True` is returned only when a warehouse adapter applied
    `where`/`sample` at the remote source itself. For all other inputs
    (paths, in-memory frames, DuckDB-extension warehouse sources), the
    caller's outer `WHERE`/`USING SAMPLE` wrapper does the work.
    """
    if isinstance(data, str | Path):
        text = str(data)
        # URI takes precedence over path: a string like
        # "s3://bucket/x.parquet" should not be treated as a relative path.
        from biopsy.warehouse._base import parse_warehouse_uri

        parsed = parse_warehouse_uri(text)
        if parsed is not None:
            scan_expr, qualified, display, pushed_down, cleanup = _scan_from_uri(
                con,
                parsed,
                credentials_env=credentials_env,
                where=where,
                sample=sample,
            )
            return scan_expr, None, display, qualified, pushed_down, cleanup

        path = Path(data).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return _scan_expr(path), path, path.name, None, False, None

    if isinstance(data, duckdb.DuckDBPyRelation):
        _materialize_relation(con, data)
        return REGISTERED_INPUT_VIEW, None, source_name or "dataframe", None, False, None

    try:
        con.register(REGISTERED_INPUT_VIEW, data)
    except (duckdb.Error, TypeError, AttributeError) as exc:
        # DuckDB raises duckdb.Error for objects it can't introspect; pandas /
        # polars / Arrow surface AttributeError or TypeError when their
        # __arrow_c_stream__ / __dataframe__ protocol probe fails. Any of
        # these means "we don't know how to read this object" — rewrap as a
        # caller-friendly TypeError.
        raise TypeError(
            "Unsupported input for biopsy.profile(). Pass a file path or a "
            "DuckDB-registerable table such as a pandas DataFrame, Polars "
            "DataFrame/LazyFrame, Arrow table, or DuckDB relation."
        ) from exc
    return REGISTERED_INPUT_VIEW, None, source_name or "dataframe", None, False, None


def _scan_from_uri(
    con: duckdb.DuckDBPyConnection,
    parsed: Any,
    *,
    credentials_env: str | None = None,
    where: list[str] | None = None,
    sample: int | None = None,
) -> tuple[str, str, str, bool, Any | None]:
    """Dispatch a parsed URI to its scheme adapter.

    DuckDB-extension adapters (object-store, Postgres) return a SQL scan
    expression DuckDB can read directly; biopsy's outer wrapper applies
    push-down via DuckDB. Arrow-via-vendor-client adapters (Snowflake,
    BigQuery) build the remote SELECT themselves with WHERE / LIMIT
    baked in, return an Arrow table, and signal `pushed_down=True` so
    biopsy doesn't re-apply.
    """
    from biopsy.warehouse._base import ScanOptions

    options = ScanOptions(
        where_sql=list(where or []),
        limit=sample,
        credentials_prefix=credentials_env,
    )
    scheme = parsed.scheme
    if scheme in {"s3", "s3a", "https", "http", "gs", "gcs"}:
        from biopsy.warehouse.object_store import open_object_store

        result = open_object_store(con, parsed, credentials_env=credentials_env)
    elif scheme in {"postgres", "postgresql"}:
        from biopsy.warehouse.postgres import open_postgres

        result = open_postgres(con, parsed, credentials_env=credentials_env)
    elif scheme == "bigquery":
        from biopsy.warehouse.bigquery import open_bigquery

        result = open_bigquery(con, parsed, options=options)
    elif scheme == "snowflake":
        from biopsy.warehouse.snowflake import open_snowflake

        result = open_snowflake(con, parsed, options=options)
    else:
        raise NotImplementedError(f"Adapter for scheme '{scheme}' is not yet implemented.")

    display = _display_for_uri(parsed)
    if result.scan_sql is not None:
        return (
            result.scan_sql,
            result.qualified_name,
            display,
            result.pushed_down,
            result.cleanup,
        )
    if result.arrow_table is not None:
        con.register(REGISTERED_INPUT_VIEW, result.arrow_table)
        return (
            REGISTERED_INPUT_VIEW,
            result.qualified_name,
            display,
            result.pushed_down,
            result.cleanup,
        )
    raise RuntimeError(f"Adapter for '{scheme}' returned neither scan_sql nor arrow_table.")


def _display_for_uri(parsed: Any) -> str:
    """Friendly short name for an URI — last path segment, or host."""
    path = parsed.path.rstrip("/")
    if path:
        return path.rsplit("/", 1)[-1]
    return parsed.host or parsed.qualified


def _materialize_relation(
    con: duckdb.DuckDBPyConnection,
    relation: duckdb.DuckDBPyRelation,
) -> None:
    # Cross-connection relations: round-trip via Arrow so DuckDB ingests
    # zero-copy instead of materializing Python tuples per row.
    arrow_table = relation.arrow()
    tmp = "__biopsy_arrow_input__"
    con.register(tmp, arrow_table)
    try:
        con.execute(f"CREATE TABLE {REGISTERED_INPUT_VIEW} AS SELECT * FROM {tmp}")
    finally:
        con.unregister(tmp)


# --- filter expression parsing -------------------------------------------


_FILTER_OPS = {
    "==": ("=", "scalar"),
    "!=": ("<>", "scalar"),
    ">=": (">=", "scalar"),
    "<=": ("<=", "scalar"),
    ">": (">", "scalar"),
    "<": ("<", "scalar"),
    "is not null": ("IS NOT NULL", "none"),
    "is null": ("IS NULL", "none"),
    "not in": ("NOT IN", "list"),
    "in": ("IN", "list"),
}

# Symbolic ops (unambiguous, can appear anywhere) are tried first, in length
# order so >= beats >. Keyword ops (English words) are tried last with strict
# word-boundary matching so values containing the word 'in' or 'is' don't
# hijack the parse.
_SYMBOLIC_OPS = ["==", "!=", ">=", "<=", ">", "<"]
_KEYWORD_OPS = ["is not null", "is null", "not in", "in"]


def parse_filter_expr(expr: str, dtypes: dict[str, str]) -> str:
    """Translate a user-facing filter expression to a SQL WHERE clause.

    Supported forms (column-first):
      - `segment in train,test,holdout`
      - `label not in unknown,other`
      - `value > 0`
      - `label == positive`
      - `event_time >= 2025-01-01`
      - `event_time is not null`

    Values are auto-typed by the column's declared dtype (numeric vs string).
    Strings are auto-quoted; commas separate list-op values (values cannot
    contain commas — quote them out of band if needed).
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty filter expression.")

    # Pass 1: symbolic operators (==, !=, >=, <=, >, <). Leftmost-symbolic-op
    # wins — but we try the longest forms first within the leftmost cluster so
    # >= beats >.
    leftmost = _find_leftmost_symbolic(expr)
    if leftmost is not None:
        idx, op_text = leftmost
        col = expr[:idx].strip()
        val = expr[idx + len(op_text) :].strip()
        if col:
            sql_op, kind = _FILTER_OPS[op_text]
            return _format_clause(col, sql_op, kind, val, dtypes)

    # Pass 2: keyword operators (is not null, is null, not in, in). Must be
    # surrounded by word boundaries AND flanked by whitespace (or end-of-string)
    # so values containing 'in' don't trigger the IN op.
    for op_text in _KEYWORD_OPS:
        pattern = rf"(?:^|\s){re.escape(op_text)}(?=\s|$)"
        m = re.search(pattern, expr, flags=re.IGNORECASE)
        if not m:
            continue
        # The match starts with optional whitespace; the op itself begins
        # after that whitespace.
        op_start = m.start() + (1 if m.group().startswith((" ", "\t")) else 0)
        col = expr[:op_start].strip()
        val = expr[m.end() :].strip()
        if not col:
            continue
        sql_op, kind = _FILTER_OPS[op_text]
        return _format_clause(col, sql_op, kind, val, dtypes)

    raise ValueError(
        f"Unrecognized filter expression: '{expr}'. "
        "Use forms like 'col in a,b,c', 'col >= 5', 'col is not null'."
    )


def _find_leftmost_symbolic(expr: str) -> tuple[int, str] | None:
    """Find the leftmost symbolic operator in `expr`. When two ops start at
    the same index, prefer the longer one (>= beats >, == beats =, etc.).
    Skip operators that occur inside a quoted segment."""
    best: tuple[int, str] | None = None
    for op in _SYMBOLIC_OPS:
        # naive scan; expr is short
        i = 0
        while True:
            idx = expr.find(op, i)
            if idx < 0:
                break
            if _inside_quotes(expr, idx):
                i = idx + 1
                continue
            if best is None or idx < best[0] or (idx == best[0] and len(op) > len(best[1])):
                best = (idx, op)
            # Take this op's leftmost occurrence; length tie-break happens above.
            break
    return best


def _inside_quotes(expr: str, idx: int) -> bool:
    """True if `expr[idx]` falls inside a single- or double-quoted segment.

    Doubled quotes (`''`, `""`) are treated as embedded literals and don't
    toggle the quote state.
    """
    in_single = False
    in_double = False
    i = 0
    while i < idx:
        c = expr[i]
        if in_single:
            if c == "'":
                if i + 1 < len(expr) and expr[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
        elif in_double:
            if c == '"':
                if i + 1 < len(expr) and expr[i + 1] == '"':
                    i += 2
                    continue
                in_double = False
        else:
            if c == "'":
                in_single = True
            elif c == '"':
                in_double = True
        i += 1
    return in_single or in_double


def _format_clause(
    col: str,
    sql_op: str,
    kind: str,
    val: str,
    dtypes: dict[str, str],
) -> str:
    if col not in dtypes:
        available = list(dtypes)[:8]
        suffix = "..." if len(dtypes) > len(available) else ""
        raise ValueError(f"Unknown column in filter: {col!r}. Available: {available}{suffix}")
    qcol = _quote_ident(col)
    col_dtype = dtypes[col]
    col_kind = kind_of(col_dtype)
    is_numeric = col_kind == "numeric"

    if kind == "none":
        if val:
            raise ValueError(f"Operator '{sql_op}' takes no value but got '{val}'.")
        return f"{qcol} {sql_op}"

    if kind == "list":
        items = [_lit(v.strip(), is_numeric) for v in val.split(",") if v.strip()]
        if not items:
            raise ValueError(f"List operator needs at least one value: '{val}'.")
        return f"{qcol} {sql_op} ({', '.join(items)})"

    # scalar
    return f"{qcol} {sql_op} {_lit(val, is_numeric)}"


def _lit(value: str, is_numeric: bool) -> str:
    """Quote a literal for SQL based on the target column's numeric-ness."""
    value = value.strip()
    # strip wrapping quotes if user supplied them
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    if is_numeric:
        # try float parse; let it fall through to quoted-string if not numeric
        try:
            float(value)
            return value
        except ValueError:
            pass
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
}
TEMPORAL_TYPES = {
    "DATE",
    "TIME",
    "TIMESTAMP",
    "TIMESTAMP_S",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMP WITH TIME ZONE",
}
BOOL_TYPES = {"BOOLEAN"}


def kind_of(dtype: str) -> str:
    """Coarse kind: numeric | temporal | bool | text | other."""
    base = dtype.split("(")[0].strip().upper()
    if base in NUMERIC_TYPES or base.startswith("DECIMAL"):
        return "numeric"
    if base in TEMPORAL_TYPES or base.startswith("TIMESTAMP"):
        return "temporal"
    if base in BOOL_TYPES:
        return "bool"
    if base in {"VARCHAR", "TEXT", "STRING", "CHAR"}:
        return "text"
    return "other"
