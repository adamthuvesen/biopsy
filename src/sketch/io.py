"""DuckDB-backed loading for CSV / Parquet."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class Source:
    """A loaded dataset, exposed as a DuckDB view named `data`."""

    con: duckdb.DuckDBPyConnection
    path: Path
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]

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
    path: str | Path,
    sample: int | None = None,
    exclude: list[str] | None = None,
    where: list[str] | None = None,
) -> Source:
    """Open a file as a DuckDB view named `data`.

    Parameters
    ----------
    sample: random reservoir sample size (deterministic seed=42).
    exclude: column names to drop from the view (e.g. known target proxies).
    where: filter expressions, AND-ed together. See `parse_filter_expr`.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    con = duckdb.connect(":memory:")
    scan = _scan_expr(path)

    # First materialize the raw scan so we know the schema for filter parsing
    # and exclusion validation.
    raw_info = con.execute(f"DESCRIBE SELECT * FROM {scan} LIMIT 0").fetchall()
    raw_columns = [r[0] for r in raw_info]
    raw_dtypes = {r[0]: r[1] for r in raw_info}

    select_clause = "*"
    if exclude:
        missing = [c for c in exclude if c not in raw_columns]
        if missing:
            raise ValueError(
                f"--exclude column(s) not in dataset: {missing}. "
                f"Available: {raw_columns[:8]}..."
            )
        excl = ", ".join(_quote_ident(c) for c in exclude)
        select_clause = f"* EXCLUDE ({excl})"

    where_clause = ""
    if where:
        parts = [parse_filter_expr(expr, raw_dtypes) for expr in where]
        where_clause = " WHERE " + " AND ".join(f"({p})" for p in parts)

    if sample:
        # CRITICAL: USING SAMPLE applies to the source *before* WHERE if they're
        # in the same SELECT, so a filtered sample collapses to ~sample × fraction
        # rows. Wrap the filtered SELECT in a subquery so the sample runs on the
        # already-filtered data.
        con.execute(
            f"CREATE VIEW data AS "
            f"SELECT * FROM (SELECT {select_clause} FROM {scan}{where_clause}) "
            f"USING SAMPLE {sample} ROWS (reservoir, 42)"
        )
    else:
        con.execute(
            f"CREATE VIEW data AS SELECT {select_clause} FROM {scan}{where_clause}"
        )

    info = con.execute("DESCRIBE data").fetchall()
    columns = [r[0] for r in info]
    dtypes = {r[0]: r[1] for r in info}
    n_rows = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]

    return Source(
        con=con,
        path=path,
        n_rows=n_rows,
        n_cols=len(columns),
        columns=columns,
        dtypes=dtypes,
    )


# --- filter expression parsing -------------------------------------------


_FILTER_OPS = {
    "==":        ("=",       "scalar"),
    "!=":        ("<>",      "scalar"),
    ">=":        (">=",      "scalar"),
    "<=":        ("<=",      "scalar"),
    ">":         (">",       "scalar"),
    "<":         ("<",       "scalar"),
    "is not null": ("IS NOT NULL", "none"),
    "is null":   ("IS NULL", "none"),
    "not in":    ("NOT IN",  "list"),
    "in":        ("IN",      "list"),
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
      - `plan not in free,trial`
      - `value > 0`
      - `status == active`
      - `signup_date >= 2025-01-01`
      - `email is not null`

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
        val = expr[idx + len(op_text):].strip()
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
        val = expr[m.end():].strip()
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
    best_idx: int | None = None
    best_op: str | None = None
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
            if best_idx is None or idx < best_idx or (idx == best_idx and len(op) > len(best_op)):  # type: ignore[arg-type]
                best_idx = idx
                best_op = op
            break  # take leftmost occurrence of this op; longer ops handled by length tie-break above
    if best_idx is None or best_op is None:
        return None
    return best_idx, best_op


def _inside_quotes(expr: str, idx: int) -> bool:
    """True if `expr[idx]` falls inside a single- or double-quoted segment."""
    single = 0
    double = 0
    for i in range(idx):
        c = expr[i]
        if c == "'" and double % 2 == 0:
            single += 1
        elif c == '"' and single % 2 == 0:
            double += 1
    return single % 2 == 1 or double % 2 == 1


def _format_clause(
    col: str, sql_op: str, kind: str, val: str, dtypes: dict[str, str],
) -> str:
    qcol = _quote_ident(col)
    col_dtype = dtypes.get(col, "VARCHAR")
    col_kind = kind_of(col_dtype)
    is_numeric = col_kind == "numeric"

    if kind == "none":
        if val:
            raise ValueError(
                f"Operator '{sql_op}' takes no value but got '{val}'."
            )
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
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
    "FLOAT", "DOUBLE", "DECIMAL",
}
TEMPORAL_TYPES = {"DATE", "TIME", "TIMESTAMP", "TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS",
                  "TIMESTAMP WITH TIME ZONE"}
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
