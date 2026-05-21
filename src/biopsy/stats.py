"""Per-column statistics computed via DuckDB SQL aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from biopsy.io import Source, kind_of
from biopsy.io import _quote_ident as _quote

_NUMERIC_TYPES: tuple[type, ...] = (int, float, np.integer, np.floating)


@dataclass
class ColumnStats:
    name: str
    dtype: str
    kind: str  # numeric | temporal | bool | text | other
    n: int
    n_null: int
    n_unique: int
    null_rate: float

    # numeric-only
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p01: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p99: float | None = None
    max: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    n_zero: int | None = None
    n_negative: int | None = None
    n_outliers_iqr: int | None = None

    # categorical/text
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    avg_len: float | None = None
    max_len: int | None = None

    # histogram (numeric) or value counts (categorical)
    histogram: list[tuple[float, float, int]] = field(default_factory=list)  # (lo, hi, count)

    # temporal histogram: list of (bucket_label, count), bucketed by month or year
    temporal_buckets: list[tuple[str, int]] = field(default_factory=list)

    @property
    def unique_rate(self) -> float:
        denom = self.n - self.n_null
        return self.n_unique / denom if denom else 0.0

    @property
    def is_constant(self) -> bool:
        return self.n_unique <= 1

    @property
    def is_near_constant(self) -> bool:
        if not self.top_values or self.n - self.n_null == 0:
            return False
        top_count = self.top_values[0][1]
        # Temporal columns store (min/max, datestr) pairs in top_values; skip.
        if not isinstance(top_count, _NUMERIC_TYPES):
            return False
        return top_count / (self.n - self.n_null) > 0.99


def compute_column(
    src: Source,
    name: str,
    hist_bins: int = 24,
    *,
    _base_counts: tuple[int, int, int] | None = None,
) -> ColumnStats:
    col = _quote(name)
    dtype = src.dtypes[name]
    kind = kind_of(dtype)

    if _base_counts is not None:
        n, n_nonnull, n_unique = _base_counts
    else:
        base = src.con.execute(
            f"SELECT COUNT(*), COUNT({col}), COUNT(DISTINCT {col}) FROM data"
        ).fetchone()
        n, n_nonnull, n_unique = base
    n_null = n - n_nonnull

    stats = ColumnStats(
        name=name,
        dtype=dtype,
        kind=kind,
        n=n,
        n_null=n_null,
        n_unique=n_unique,
        null_rate=n_null / n if n else 0.0,
    )

    if kind == "numeric" and n_nonnull > 0:
        # One scan: aggregates + quantiles + IQR-outlier count + sign counts.
        # The CTE binds the quantile vector once so the outlier predicate can
        # reference it without recomputing.
        row = src.con.execute(f"""
            WITH q AS (
                SELECT quantile_cont({col}, [0.01, 0.25, 0.5, 0.75, 0.99]) AS qs
                FROM data
            ),
            bounds AS (
                SELECT
                    qs[2] AS q25, qs[4] AS q75,
                    qs[2] - 1.5 * (qs[4] - qs[2]) AS lo,
                    qs[4] + 1.5 * (qs[4] - qs[2]) AS hi
                FROM q
            )
            SELECT
                AVG({col})::DOUBLE,
                STDDEV_SAMP({col})::DOUBLE,
                MIN({col})::DOUBLE,
                (SELECT qs[1] FROM q)::DOUBLE,
                (SELECT q25 FROM bounds)::DOUBLE,
                (SELECT qs[3] FROM q)::DOUBLE,
                (SELECT q75 FROM bounds)::DOUBLE,
                (SELECT qs[5] FROM q)::DOUBLE,
                MAX({col})::DOUBLE,
                skewness({col})::DOUBLE,
                kurtosis({col})::DOUBLE,
                SUM(CASE WHEN {col} = 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN {col} < 0 THEN 1 ELSE 0 END),
                SUM(CASE
                    WHEN (SELECT q75 - q25 FROM bounds) > 0
                     AND ({col} < (SELECT lo FROM bounds) OR {col} > (SELECT hi FROM bounds))
                    THEN 1 ELSE 0 END)
            FROM data
        """).fetchone()
        (stats.mean, stats.std, stats.min, stats.p01, stats.p25, stats.p50,
         stats.p75, stats.p99, stats.max, stats.skew, stats.kurtosis,
         stats.n_zero, stats.n_negative, stats.n_outliers_iqr) = row
        top = src.con.execute(f"""
            SELECT {col}::VARCHAR, COUNT(*) AS c
            FROM data
            WHERE {col} IS NOT NULL
            GROUP BY 1
            ORDER BY c DESC
            LIMIT 1
        """).fetchall()
        stats.top_values = [(v, c) for v, c in top]
        # skew/kurtosis are undefined for constant or near-constant columns
        if stats.std is None or stats.std == 0:
            stats.skew = None
            stats.kurtosis = None

        stats.histogram = _numeric_histogram(
            src, name, hist_bins, lo=stats.min, hi=stats.max,
        )

    elif kind in {"text", "bool", "other"} and n_nonnull > 0:
        top = src.con.execute(f"""
            SELECT {col}::VARCHAR, COUNT(*) AS c
            FROM data
            WHERE {col} IS NOT NULL
            GROUP BY 1
            ORDER BY c DESC
            LIMIT 10
        """).fetchall()
        stats.top_values = [(v, c) for v, c in top]
        stats.histogram = [(i, i + 1, c) for i, (_v, c) in enumerate(stats.top_values)]

        if kind == "text":
            lens = src.con.execute(
                f"SELECT AVG(LENGTH({col}))::DOUBLE, MAX(LENGTH({col})) "
                f"FROM data WHERE {col} IS NOT NULL"
            ).fetchone()
            stats.avg_len, stats.max_len = lens

    elif kind == "temporal" and n_nonnull > 0:
        row = src.con.execute(f"""
            SELECT MIN({col})::VARCHAR, MAX({col})::VARCHAR,
                   epoch(MAX({col}))::BIGINT - epoch(MIN({col}))::BIGINT
            FROM data WHERE {col} IS NOT NULL
        """).fetchone()
        stats.top_values = [("min", row[0]), ("max", row[1])]
        span_seconds = row[2] or 0
        # Choose bucket: month if span < 5 years, year otherwise; day if < 60 days.
        if span_seconds < 60 * 86400:
            trunc = "day"
            fmt = "%Y-%m-%d"
        elif span_seconds < 5 * 365 * 86400:
            trunc = "month"
            fmt = "%Y-%m"
        else:
            trunc = "year"
            fmt = "%Y"
        buckets = src.con.execute(f"""
            SELECT strftime(date_trunc('{trunc}', {col}), '{fmt}') AS bucket,
                   COUNT(*) AS c
            FROM data WHERE {col} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """).fetchall()
        stats.temporal_buckets = [(b, int(c)) for b, c in buckets]

    return stats


def _numeric_histogram(
    src: Source,
    name: str,
    bins: int,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> list[tuple[float, float, int]]:
    """Equal-width histogram via DuckDB.

    Falls back to a single bin when min == max. NaN doubles are excluded so
    bin counts sum to `n_nonnull`.
    """
    col = _quote(name)
    where = f"{col} IS NOT NULL AND NOT isnan({col}::DOUBLE)"
    if lo is None or hi is None:
        row = src.con.execute(
            f"SELECT MIN({col})::DOUBLE, MAX({col})::DOUBLE FROM data WHERE {where}"
        ).fetchone()
        lo, hi = row
    if lo is None or hi is None:
        return []
    if lo == hi:
        c = src.con.execute(
            f"SELECT COUNT(*) FROM data WHERE {col} = ?", [lo]
        ).fetchone()[0]
        return [(lo, hi, c)]

    width = (hi - lo) / bins
    # bucket via floor((x - lo) / width), clipped to [0, bins-1]; GREATEST
    # guards floating-point underflow at x == lo producing a negative bin.
    rows = src.con.execute(f"""
        WITH b AS (
            SELECT GREATEST(
                0,
                LEAST(CAST(FLOOR(({col}::DOUBLE - ?) / ?) AS INTEGER), ?)
            ) AS bin
            FROM data WHERE {where}
        )
        SELECT bin, COUNT(*) FROM b GROUP BY bin ORDER BY bin
    """, [lo, width, bins - 1]).fetchall()
    counts = {int(b): c for b, c in rows}
    return [
        (lo + i * width, lo + (i + 1) * width, counts.get(i, 0))
        for i in range(bins)
    ]


def compute_all(src: Source, hist_bins: int = 24) -> dict[str, ColumnStats]:
    """Compute per-column stats.

    The base counts (COUNT(*), COUNT(col), COUNT(DISTINCT col)) are batched
    into a single DuckDB query across all columns, then the per-column
    enrichments (numeric aggregates, histograms, top values) follow.
    """
    base_counts = _batched_base_counts(src)
    out: dict[str, ColumnStats] = {}
    for name in src.columns:
        out[name] = compute_column(src, name, hist_bins, _base_counts=base_counts.get(name))
    return out


def _batched_base_counts(src: Source) -> dict[str, tuple[int, int, int]]:
    """One SQL pass for `COUNT(*), COUNT(col), COUNT(DISTINCT col)` over
    every column. Returns `{column: (n, n_nonnull, n_unique)}`.
    """
    if not src.columns:
        return {}
    selects = ["COUNT(*)"]
    for c in src.columns:
        q = _quote(c)
        selects.append(f"COUNT({q})")
        selects.append(f"COUNT(DISTINCT {q})")
    row = src.con.execute(f"SELECT {', '.join(selects)} FROM data").fetchone()
    n = int(row[0])
    out: dict[str, tuple[int, int, int]] = {}
    for i, col in enumerate(src.columns):
        nonnull = int(row[1 + 2 * i])
        nunique = int(row[2 + 2 * i])
        out[col] = (n, nonnull, nunique)
    return out
