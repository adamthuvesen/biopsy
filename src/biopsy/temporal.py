"""Temporal leakage and drift detection.

When a dataset has a time column, profile features for:
- Predictive-power gap between random-CV split and time-ordered split (the
  classic "future-knowing" leak).
- Distribution drift between early and late halves (concept shift).
- Time-monotonic features (row counters, ingest timestamps masquerading as
  features, cumulative quantities that include future).
- Target rate drift across time deciles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

from biopsy.correlations import _spearman, _valid_mask, pps
from biopsy.io import Source
from biopsy.matrix import _fetch_object_array
from biopsy.stats import ColumnStats, _quote
from biopsy.targets import target_task_kind

# --- thresholds (single source of truth) ----------------------------------

LEAK_GAP_THRESHOLD = 0.25          # random_pps − time_pps must exceed this
LEAK_MIN_RANDOM_PPS = 0.35         # only flag if the feature is actually predictive
POST_EVENT_MIN_RANDOM_PPS = 0.30
POST_EVENT_MAX_TIME_PPS = 0.05
POST_EVENT_MIN_GAP = 0.25
POST_EVENT_MIN_DRIFT_KS = 0.15

LeakageKind = Literal[
    "none",
    "random_cv",
    "post_event",
    "drift",
    "strong_drift",
    "monotonic",
]
DRIFT_KS_THRESHOLD = 0.3
DRIFT_MIN_PPS = 0.1
STRONG_DRIFT_KS = 0.5              # flag drift at info level even without PPS signal
MONOTONIC_THRESHOLD = 0.95
TARGET_DRIFT_RATIO = 2.0
TARGET_DRIFT_BINARY_DIFF = 0.2     # 20 percentage points
TARGET_DRIFT_REGRESSION_DIFF_SCALE = 1.0  # mean range >= 1 target-IQR/std
MIN_ROWS = 1000
MIN_TIME_VALUES = 10
MAX_EXACT_TIME_BUCKETS = 14
TEST_FRACTION = 0.3
MIN_TEST_POSITIVES_CLASSIF = 30


# --- dataclasses -----------------------------------------------------------

@dataclass
class TemporalSignal:
    feature: str
    time_pps: float | None
    random_pps: float | None
    drift_ks: float | None
    time_monotonicity: float | None
    severity: str           # "critical" | "warning" | "info" | "none"
    reason: str             # human-readable
    leakage_kind: LeakageKind = "none"

    @property
    def leak_gap(self) -> float | None:
        if self.random_pps is None or self.time_pps is None:
            return None
        return self.random_pps - self.time_pps


@dataclass
class TemporalReport:
    time_column: str
    target: str | None
    signals: list[TemporalSignal]
    # binary: max-min rate; regression_ratio: max/min; regression_diff: max-min
    target_drift: float | None
    target_drift_kind: str | None
    insufficient: str | None       # reason for skipping per-feature analysis, if any
    target_drift_score: float | None = None  # threshold-scale score; same units except diff
    time_buckets: list[TimeBucket] = field(default_factory=list)


@dataclass
class TimeBucket:
    label: str
    n_rows: int
    n_target: int | None = None
    target_rate: float | None = None
    target_mean: float | None = None


# --- time column resolution ------------------------------------------------

def resolve_time_column(
    stats: dict[str, ColumnStats],
    explicit: str | None,
) -> tuple[str | None, str | None]:
    """Return (resolved_column, info_message).

    info_message is set when we want the caller to surface a finding (e.g.
    "multiple temporal columns — pass --time").
    """
    if explicit is not None:
        if explicit not in stats:
            return None, f"Time column '{explicit}' not found in dataset."
        if stats[explicit].kind != "temporal":
            return None, (
                f"Time column '{explicit}' is {stats[explicit].dtype}, not temporal. "
                "Pick a DATE/TIMESTAMP column or parse it before profiling."
            )
        return explicit, None

    temporals = [name for name, s in stats.items() if s.kind == "temporal"]
    if len(temporals) == 1:
        return temporals[0], None
    if len(temporals) > 1:
        return None, (
            f"Multiple datetime columns found ({', '.join(temporals)}) — "
            "pass --time to enable temporal analysis."
        )
    return None, None


# --- main entry point ------------------------------------------------------

def temporal_signals(
    src: Source,
    stats: dict[str, ColumnStats],
    time_col: str,
    target: str | None = None,
    max_rows: int = 50_000,
) -> TemporalReport | None:
    """Run the full temporal analysis. Returns None if preconditions fail."""

    if src.n_rows < MIN_ROWS:
        return None

    t_stats = stats.get(time_col)
    if t_stats is None or t_stats.kind != "temporal":
        return None
    if t_stats.n_unique < MIN_TIME_VALUES:
        buckets = _time_bucket_summary(src, stats, time_col, target)
        target_drift, target_drift_kind, target_drift_score = _target_drift_from_buckets(
            buckets, stats.get(target) if target else None,
        )
        return TemporalReport(
            time_column=time_col, target=target, signals=[],
            target_drift=target_drift,
            target_drift_kind=target_drift_kind,
            target_drift_score=target_drift_score,
            time_buckets=buckets,
            insufficient=(
                f"Only {t_stats.n_unique} distinct value(s) in `{time_col}` — "
                f"need ≥{MIN_TIME_VALUES} for leakage-style time-ordered splits. "
                "Target-by-period drift was still computed."
            ),
        )

    # Determine target kind early so we can pull the target alongside features.
    target_kind: str | None = None
    if target and target in stats:
        target_kind = target_task_kind(stats[target])

    # Pick the feature set — same eligibility as target_signal.
    features = [
        name for name, s in stats.items()
        if name not in {time_col, target}
        and not s.is_constant
        and (s.kind == "numeric" or (s.kind in {"text", "bool"} and s.n_unique <= 100))
    ]
    if not features:
        return TemporalReport(
            time_column=time_col, target=target, signals=[],
            target_drift=None, target_drift_kind=None, target_drift_score=None,
            time_buckets=_time_bucket_summary(src, stats, time_col, target),
            insufficient="No eligible features for temporal analysis.",
        )

    cols_to_pull = features + ([target] if target else []) + [time_col]
    quoted = ", ".join(_quote(c) for c in cols_to_pull)

    # Filter missing timestamps before sampling; DuckDB applies USING SAMPLE to
    # the relation it is attached to.
    raw = _fetch_object_array(
        src.con,
        f"""
        SELECT {quoted}
        FROM (
            SELECT {quoted} FROM data
            WHERE {_quote(time_col)} IS NOT NULL
        ) USING SAMPLE {max_rows} ROWS (reservoir, 42)
        ORDER BY {_quote(time_col)}
        """,
        cols_to_pull,
    )
    if raw.shape[0] < MIN_ROWS:
        return None

    time_values = raw[:, -1]
    time_as_float = _time_to_float(time_values)
    n = len(raw)
    split_point = int(n * (1 - TEST_FRACTION))
    early_idx = np.arange(split_point)
    late_idx = np.arange(split_point, n)

    # Random split with the same proportions for fair comparison.
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    random_train_idx = perm[:split_point]
    random_test_idx = perm[split_point:]

    # Target prep
    y_full: np.ndarray | None = None
    target_valid: np.ndarray | None = None
    if target is not None and target_kind is not None:
        t_col_idx = len(features)
        y_raw = raw[:, t_col_idx]
        target_valid = _valid_mask(y_raw)
        if target_kind == "classification":
            y_full = np.full(n, -1, dtype=int)
            y_full[target_valid] = LabelEncoder().fit_transform(
                y_raw[target_valid].astype(str)
            ).astype(int)
        else:
            y_full = np.full(n, np.nan, dtype=np.float64)
            y_full[target_valid] = np.asarray(y_raw[target_valid], dtype=np.float64)

    # Per-feature analysis
    def _run_feature(j: int, feat: str) -> TemporalSignal | None:
        feat_valid = _valid_mask(raw[:, j])
        return _analyze_feature(
            feat=feat,
            kind=stats[feat].kind,
            x_raw=raw[:, j],
            feat_valid=feat_valid,
            time_as_float=time_as_float,
            split_point=split_point,
            early_idx=early_idx,
            late_idx=late_idx,
            random_train_idx=random_train_idx,
            random_test_idx=random_test_idx,
            y_full=y_full,
            target_valid=target_valid,
            target_kind=target_kind,
            n_unique=stats[feat].n_unique,
            n_rows=stats[feat].n,
            n_nulls=stats[feat].n_null,
        )

    # Parallelize the per-feature loop when it's wide enough to amortize
    # joblib's process-fork cost. Each call fits up to two decision trees,
    # which is CPU-bound and embarrassingly parallel.
    results: list[TemporalSignal | None]
    if len(features) >= 16:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_run_feature)(j, feat) for j, feat in enumerate(features)
        )
    else:
        results = [_run_feature(j, feat) for j, feat in enumerate(features)]

    signals: list[TemporalSignal] = []
    monotonic_flags = 0
    for feat_signal in results:
        if feat_signal is None:
            continue
        if feat_signal.time_monotonicity and feat_signal.time_monotonicity >= MONOTONIC_THRESHOLD:
            monotonic_flags += 1
        signals.append(feat_signal)

    # If most features are time-monotonic, the time column is probably ingest
    # order, not event time — demote all monotonic flags to a single warning.
    if monotonic_flags > max(2, len(signals) // 2):
        for s in signals:
            if (
                s.severity == "warning"
                and s.time_monotonicity
                and s.time_monotonicity >= MONOTONIC_THRESHOLD
            ):
                s.severity = "info"
                s.reason = f"Time-monotonic, but `{time_col}` looks like ingest order."

    # Target drift
    target_drift, target_drift_kind, target_drift_score = _target_drift(
        y_full, target_valid, target_kind, time_as_float,
    ) if y_full is not None else (None, None, None)

    return TemporalReport(
        time_column=time_col,
        target=target,
        signals=signals,
        target_drift=target_drift,
        target_drift_kind=target_drift_kind,
        target_drift_score=target_drift_score,
        time_buckets=_time_bucket_summary(src, stats, time_col, target),
        insufficient=None,
    )


# --- per-feature analysis --------------------------------------------------

def _analyze_feature(
    feat: str,
    kind: str,
    x_raw: np.ndarray,
    feat_valid: np.ndarray,
    time_as_float: np.ndarray,
    split_point: int,
    early_idx: np.ndarray,
    late_idx: np.ndarray,
    random_train_idx: np.ndarray,
    random_test_idx: np.ndarray,
    y_full: np.ndarray | None,
    target_valid: np.ndarray | None,
    target_kind: str | None,
    n_unique: int,
    n_rows: int,
    n_nulls: int,
) -> TemporalSignal | None:
    n = len(x_raw)
    if feat_valid.sum() < MIN_ROWS // 2:
        return None

    # Encode full column (NaN for invalid rows; we mask below).
    x_enc = np.full(n, np.nan, dtype=np.float64)
    if kind == "numeric":
        try:
            x_enc[feat_valid] = np.asarray(x_raw[feat_valid], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    else:
        encoded = LabelEncoder().fit_transform(x_raw[feat_valid].astype(str)).astype(np.float64)
        x_enc[feat_valid] = encoded

    drift_ks = _drift_stat(
        x_raw[:split_point][feat_valid[:split_point]],
        x_raw[split_point:][feat_valid[split_point:]],
        kind=kind,
    )

    # Time monotonicity — |Spearman(feature, time)| on rows where both are valid.
    time_monotonicity = None
    valid_for_mono = feat_valid
    if valid_for_mono.sum() >= 30:
        spear = _spearman(x_enc[valid_for_mono], time_as_float[valid_for_mono])
        time_monotonicity = abs(spear) if spear is not None else None

    # PPS gap requires a target.
    time_pps: float | None = None
    random_pps: float | None = None
    if y_full is not None and target_valid is not None and target_kind is not None:
        joint_valid = feat_valid & target_valid

        # time-ordered split (train=early, test=late) restricted to joint-valid rows
        train_mask = joint_valid.copy()
        train_mask[late_idx] = False
        test_mask = joint_valid.copy()
        test_mask[early_idx] = False
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if (
            len(train_idx) >= 50
            and len(test_idx) >= 50
            and _enough_test_classes(y_full[test_idx], target_kind)
        ):
            time_pps = pps(
                x_enc.reshape(-1, 1), y_full, target_kind,
                split=(train_idx, test_idx),
            )

        # random split with same sizes
        rand_train = random_train_idx[joint_valid[random_train_idx]]
        rand_test = random_test_idx[joint_valid[random_test_idx]]
        if (
            len(rand_train) >= 50
            and len(rand_test) >= 50
            and _enough_test_classes(y_full[rand_test], target_kind)
        ):
            random_pps = pps(
                x_enc.reshape(-1, 1), y_full, target_kind,
                split=(rand_train, rand_test),
            )

    severity, reason, leakage_kind = _classify(
        random_pps=random_pps,
        time_pps=time_pps,
        drift_ks=drift_ks,
        time_monotonicity=time_monotonicity,
        n_unique=n_unique,
        n_nonnull=n_rows - n_nulls,
    )

    return TemporalSignal(
        feature=feat,
        time_pps=time_pps,
        random_pps=random_pps,
        drift_ks=drift_ks,
        time_monotonicity=time_monotonicity,
        severity=severity,
        reason=reason,
        leakage_kind=leakage_kind,
    )


def _classify(
    random_pps: float | None,
    time_pps: float | None,
    drift_ks: float | None,
    time_monotonicity: float | None,
    n_unique: int,
    n_nonnull: int,
) -> tuple[str, str, LeakageKind]:
    # Leakage: predictive power collapses on time-ordered eval
    if (
        random_pps is not None and time_pps is not None
        and random_pps >= LEAK_MIN_RANDOM_PPS
        and (random_pps - time_pps) >= LEAK_GAP_THRESHOLD
    ):
        return "critical", (
            f"Predicts target on random CV ({random_pps:.2f}) but fails on "
            f"time-ordered split ({time_pps:.2f}). Likely contains future information."
        ), "random_cv"

    # Post-event leakage: target signal is present in only part of the time
    # range. Lower random_pps threshold catches features that pop in late, are
    # computed from future events, and don't generalize across time.
    if (
        random_pps is not None and time_pps is not None
        and random_pps >= POST_EVENT_MIN_RANDOM_PPS
        and time_pps < POST_EVENT_MAX_TIME_PPS
        and (random_pps - time_pps) >= POST_EVENT_MIN_GAP
        and drift_ks is not None and drift_ks >= POST_EVENT_MIN_DRIFT_KS
    ):
        return "critical", (
            f"Random-CV predictive signal ({random_pps:.2f}) but time-ordered "
            f"split scores near zero ({time_pps:.2f}) with strong drift "
            f"(KS={drift_ks:.2f}). Likely contains future information from "
            "post-event values."
        ), "post_event"

    # Drift + non-trivial predictive signal
    if (
        drift_ks is not None and drift_ks >= DRIFT_KS_THRESHOLD
        and ((random_pps or 0) >= DRIFT_MIN_PPS or (time_pps or 0) >= DRIFT_MIN_PPS)
    ):
        return "warning", (
            f"Distribution shifts over time (KS={drift_ks:.2f}) on a predictive feature. "
            "Production model may degrade."
        ), "drift"

    # Strong drift without a PPS signal — common for regression targets where
    # individual features score near-zero PPS but still shift seasonally/temporally.
    if drift_ks is not None and drift_ks >= STRONG_DRIFT_KS:
        return "info", (
            f"Distribution shifts significantly over time (KS={drift_ks:.2f}). "
            "May indicate seasonal patterns or concept drift."
        ), "strong_drift"

    # Time-monotonic + unique-per-row
    if (
        time_monotonicity is not None and time_monotonicity >= MONOTONIC_THRESHOLD
        and n_unique >= n_nonnull * 0.9
    ):
        return "warning", (
            f"Strictly increases/decreases with time (Spearman={time_monotonicity:.2f}) "
            "and is unique-per-row — likely a row counter or ingest timestamp."
        ), "monotonic"

    return "none", "", "none"


def infer_leakage_kind_from_legacy(
    severity: str,
    reason: str,
) -> LeakageKind:
    """Back-fill `leakage_kind` for saved profiles emitted before the field existed."""
    if severity != "critical":
        if severity == "warning" and "Distribution shifts over time" in reason:
            return "drift"
        if severity == "warning" and "unique-per-row" in reason:
            return "monotonic"
        if severity == "info" and "Distribution shifts significantly" in reason:
            return "strong_drift"
        return "none"
    reason_l = reason.lower()
    if "post-event" in reason_l:
        return "post_event"
    if "future information" in reason_l:
        return "random_cv"
    return "none"


def temporal_signal_from_payload(payload: dict) -> TemporalSignal:
    """Build a signal from JSON, inferring `leakage_kind` when absent."""
    leakage_kind = payload.get("leakage_kind")
    if leakage_kind is None:
        leakage_kind = infer_leakage_kind_from_legacy(
            str(payload.get("severity", "none")),
            str(payload.get("reason", "")),
        )
    return TemporalSignal(
        feature=str(payload["feature"]),
        time_pps=payload.get("time_pps"),
        random_pps=payload.get("random_pps"),
        drift_ks=payload.get("drift_ks"),
        time_monotonicity=payload.get("time_monotonicity"),
        severity=str(payload.get("severity", "none")),
        reason=str(payload.get("reason", "")),
        leakage_kind=leakage_kind,
    )


def _enough_test_classes(y: np.ndarray, target_kind: str) -> bool:
    if target_kind != "classification":
        return True
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return False
    if len(classes) == 2:
        # Both classes need real support — a 5000/3 split passes f1_weighted
        # but its time_pps signal is unreliable and feeds leakage detection.
        return bool(counts.min() >= MIN_TEST_POSITIVES_CLASSIF)
    return len(y) >= MIN_TEST_POSITIVES_CLASSIF


# --- helpers ---------------------------------------------------------------

def _time_to_float(values: np.ndarray) -> np.ndarray:
    """Convert temporal values to a float scalar suitable for ranking."""
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
            continue
        try:
            parsed = np.datetime64(v).astype("datetime64[s]")
            out[i] = np.nan if np.isnat(parsed) else float(parsed.astype(np.int64))
        except (TypeError, ValueError):
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                out[i] = np.nan
    return out


def _drift_stat(early: np.ndarray, late: np.ndarray, kind: str) -> float | None:
    if len(early) < 30 or len(late) < 30:
        return None
    if kind == "numeric":
        try:
            early_f = np.asarray(early, dtype=np.float64)
            late_f = np.asarray(late, dtype=np.float64)
        except (TypeError, ValueError):
            # Values declared numeric by DuckDB occasionally include Decimal /
            # object dtypes that won't coerce — skip drift for that column.
            return None
        stat = ks_2samp(early_f, late_f, alternative="two-sided", method="auto").statistic
        return float(stat)
    # categorical / boolean: total variation distance on top-k frequencies.
    early_str = early.astype(str)
    late_str = late.astype(str)
    # Single sort+count per side via np.unique with return_counts — replaces
    # the O(K·N) per-category mean() loop with two O(N log N) passes.
    e_cats, e_counts = np.unique(early_str, return_counts=True)
    l_cats, l_counts = np.unique(late_str, return_counts=True)
    if len(set(e_cats) | set(l_cats)) > 100:
        return None
    e_total = e_counts.sum()
    l_total = l_counts.sum()
    e_lookup = dict(zip(e_cats, e_counts / e_total, strict=True))
    l_lookup = dict(zip(l_cats, l_counts / l_total, strict=True))
    tv = 0.0
    for cat in e_lookup.keys() | l_lookup.keys():
        tv += abs(e_lookup.get(cat, 0.0) - l_lookup.get(cat, 0.0))
    return float(0.5 * tv)


def _time_bucket_summary(
    src: Source,
    stats: dict[str, ColumnStats],
    time_col: str,
    target: str | None,
) -> list[TimeBucket]:
    qtime = _quote(time_col)
    use_deciles = stats[time_col].n_unique > MAX_EXACT_TIME_BUCKETS
    if target is None or target not in stats:
        if use_deciles:
            rows = src.con.execute(f"""
                WITH bucketed AS (
                    SELECT *,
                           ntile(10) OVER (ORDER BY {qtime}) AS __biopsy_bucket
                    FROM data
                    WHERE {qtime} IS NOT NULL
                )
                SELECT
                    MIN({qtime})::VARCHAR || ' - ' || MAX({qtime})::VARCHAR AS bucket,
                    COUNT(*) AS n_rows
                FROM bucketed
                GROUP BY __biopsy_bucket
                ORDER BY __biopsy_bucket
            """).fetchall()
        else:
            rows = src.con.execute(f"""
                SELECT {qtime}::VARCHAR AS bucket, COUNT(*) AS n_rows
                FROM data
                WHERE {qtime} IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """).fetchall()
        return [TimeBucket(label=str(label), n_rows=int(n_rows)) for label, n_rows in rows]

    t_stats = stats[target]
    qtarget = _quote(target)
    target_kind = target_task_kind(t_stats)
    if target_kind == "classification":
        # Pick the minority class as "positive" — that's the rare event whose
        # drift matters. Lexicographic ORDER BY 1 DESC happens to pick the
        # majority class for binary numeric targets (1 > 0), which is the
        # wrong thing.
        if not t_stats.top_values:
            return []
        positive_value = min(t_stats.top_values, key=lambda v_c: v_c[1])[0]
        pos = str(positive_value).replace("'", "''")
        target_rate_sql = f"""
            COUNT(*) AS n_rows,
            COUNT({qtarget}) AS n_target,
            AVG(
                CASE
                    WHEN {qtarget} IS NULL THEN NULL
                    WHEN {qtarget}::VARCHAR = '{pos}' THEN 1.0
                    ELSE 0.0
                END
            ) AS target_rate
        """
        if use_deciles:
            rows = src.con.execute(f"""
                WITH bucketed AS (
                    SELECT *,
                           ntile(10) OVER (ORDER BY {qtime}) AS __biopsy_bucket
                    FROM data
                    WHERE {qtime} IS NOT NULL
                )
                SELECT
                    MIN({qtime})::VARCHAR || ' - ' || MAX({qtime})::VARCHAR AS bucket,
                    {target_rate_sql}
                FROM bucketed
                GROUP BY __biopsy_bucket
                ORDER BY __biopsy_bucket
            """).fetchall()
        else:
            rows = src.con.execute(f"""
                SELECT
                    {qtime}::VARCHAR AS bucket,
                    {target_rate_sql}
                FROM data
                WHERE {qtime} IS NOT NULL
                GROUP BY 1
                ORDER BY 1
            """).fetchall()
        return [
            TimeBucket(
                label=str(label),
                n_rows=int(n_rows),
                n_target=int(n_target),
                target_rate=float(target_rate) if target_rate is not None else None,
            )
            for label, n_rows, n_target, target_rate in rows
        ]

    if use_deciles:
        rows = src.con.execute(f"""
            WITH bucketed AS (
                SELECT *,
                       ntile(10) OVER (ORDER BY {qtime}) AS __biopsy_bucket
                FROM data
                WHERE {qtime} IS NOT NULL
            )
            SELECT
                MIN({qtime})::VARCHAR || ' - ' || MAX({qtime})::VARCHAR AS bucket,
                COUNT(*) AS n_rows,
                COUNT({qtarget}) AS n_target,
                AVG({qtarget}::DOUBLE) AS target_mean
            FROM bucketed
            GROUP BY __biopsy_bucket
            ORDER BY __biopsy_bucket
        """).fetchall()
    else:
        rows = src.con.execute(f"""
            SELECT
                {qtime}::VARCHAR AS bucket,
                COUNT(*) AS n_rows,
                COUNT({qtarget}) AS n_target,
                AVG({qtarget}::DOUBLE) AS target_mean
            FROM data
            WHERE {qtime} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """).fetchall()
    return [
        TimeBucket(
            label=str(label),
            n_rows=int(n_rows),
            n_target=int(n_target),
            target_mean=float(target_mean) if target_mean is not None else None,
        )
        for label, n_rows, n_target, target_mean in rows
    ]


def _target_drift_from_buckets(
    buckets: list[TimeBucket],
    target_stats: ColumnStats | None,
) -> tuple[float | None, str | None, float | None]:
    if target_stats is None or len(buckets) < 2:
        return None, None, None
    kind = target_task_kind(target_stats)
    if kind == "classification":
        rates = [b.target_rate for b in buckets if b.target_rate is not None]
        if len(rates) < 2:
            return None, None, None
        drift = float(max(rates) - min(rates))
        # We only track the minority class in _time_bucket_summary; for K>2 the
        # bucket-derived drift only reflects that one class, so it's still
        # "binary" in shape. _target_drift covers true multiclass when enough
        # rows survive sampling.
        return drift, "binary", drift

    means = [b.target_mean for b in buckets if b.target_mean is not None]
    if len(means) < 2:
        return None, None, None
    if min(means) > 0 or max(means) < 0:
        abs_means = [abs(m) for m in means]
        drift = float(max(abs_means) / min(abs_means))
        return drift, "regression_ratio", drift
    diff = float(max(means) - min(means))
    return diff, "regression_diff", None


def _target_drift(
    y_full: np.ndarray | None,
    target_valid: np.ndarray | None,
    target_kind: str | None,
    time_as_float: np.ndarray,
) -> tuple[float | None, str | None, float | None]:
    """Drift across time deciles.

    Kinds: binary | multiclass | regression_ratio | regression_diff.
    """
    if y_full is None or target_valid is None or target_kind is None:
        return None, None, None
    mask = target_valid & ~np.isnan(time_as_float)
    if mask.sum() < 100:
        return None, None, None
    y = y_full[mask]
    t = time_as_float[mask]
    order = np.argsort(t)
    y_sorted = y[order]

    # split rows into 10 chronological deciles. mask>=100 guarantees each is >=10.
    deciles = np.array_split(y_sorted, 10)

    if target_kind == "classification":
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            pos = unique_classes.max()
            rates = np.array([(d == pos).mean() for d in deciles])
            drift = float(rates.max() - rates.min())
            return drift, "binary", drift
        max_drift = 0.0
        for c in unique_classes:
            rates = np.array([(d == c).mean() for d in deciles])
            max_drift = max(max_drift, float(rates.max() - rates.min()))
        return max_drift, "multiclass", max_drift

    # regression: ratio when means are strictly positive OR strictly negative
    # (in absolute value), otherwise fall back to scaled difference.
    means = np.array([float(d.astype(np.float64).mean()) for d in deciles])
    if means.min() > 0 or means.max() < 0:
        abs_means = np.abs(means)
        drift = float(abs_means.max() / abs_means.min())
        return drift, "regression_ratio", drift
    diff = float(means.max() - means.min())
    scale = _target_scale(y.astype(np.float64))
    score = diff / scale if scale > 0 else 0.0
    return diff, "regression_diff", score


def _target_scale(y: np.ndarray) -> float:
    q25, q75 = np.percentile(y, [25, 75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return iqr
    std = float(np.std(y))
    return std if std > 0 else 0.0


def is_target_drifted(report: TemporalReport) -> bool:
    """True if the target's distribution drifts meaningfully across deciles."""
    if report.target_drift is None or report.target_drift_kind is None:
        return False
    kind = report.target_drift_kind
    if kind == "binary":
        return report.target_drift >= TARGET_DRIFT_BINARY_DIFF
    if kind == "multiclass":
        return report.target_drift >= TARGET_DRIFT_BINARY_DIFF
    if kind == "regression_ratio":
        return report.target_drift >= TARGET_DRIFT_RATIO
    if kind == "regression_diff":
        return (report.target_drift_score or 0.0) >= TARGET_DRIFT_REGRESSION_DIFF_SCALE
    return False
