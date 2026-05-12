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

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

from sketch.correlations import _spearman, _valid_mask, pps
from sketch.io import Source
from sketch.stats import ColumnStats, _quote

# --- thresholds (single source of truth) ----------------------------------

LEAK_GAP_THRESHOLD = 0.25          # random_pps − time_pps must exceed this
LEAK_MIN_RANDOM_PPS = 0.35         # only flag if the feature is actually predictive
DRIFT_KS_THRESHOLD = 0.3
DRIFT_MIN_PPS = 0.1
MONOTONIC_THRESHOLD = 0.95
TARGET_DRIFT_RATIO = 2.0
TARGET_DRIFT_BINARY_DIFF = 0.2     # 20 percentage points
MIN_ROWS = 1000
MIN_TIME_VALUES = 10
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
        return TemporalReport(
            time_column=time_col, target=target, signals=[],
            target_drift=None, target_drift_kind=None,
            insufficient=(
                f"Only {t_stats.n_unique} distinct value(s) in `{time_col}` — "
                f"need ≥{MIN_TIME_VALUES} for a meaningful time-ordered split."
            ),
        )

    # Determine target kind early so we can pull the target alongside features.
    target_kind: str | None = None
    if target and target in stats:
        ts = stats[target]
        target_kind = (
            "classification" if (
                ts.kind in {"text", "bool"} or
                (ts.kind == "numeric" and ts.n_unique <= 20)
            ) else "regression"
        )

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
            target_drift=None, target_drift_kind=None,
            insufficient="No eligible features for temporal analysis.",
        )

    cols_to_pull = features + ([target] if target else []) + [time_col]
    quoted = ", ".join(_quote(c) for c in cols_to_pull)

    # Filter missing timestamps before sampling; DuckDB applies USING SAMPLE to
    # the relation it is attached to.
    rel = src.con.execute(f"""
        SELECT {quoted}
        FROM (
            SELECT {quoted} FROM data
            WHERE {_quote(time_col)} IS NOT NULL
        ) USING SAMPLE {max_rows} ROWS (reservoir, 42)
        ORDER BY {_quote(time_col)}
    """)
    sample = rel.fetchall()
    if len(sample) < MIN_ROWS:
        return None
    raw = np.array(sample, dtype=object)

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
    signals: list[TemporalSignal] = []
    monotonic_flags = 0

    for j, feat in enumerate(features):
        feat_valid = _valid_mask(raw[:, j])
        feat_signal = _analyze_feature(
            feat=feat,
            kind=stats[feat].kind,
            x_raw=raw[:, j],
            feat_valid=feat_valid,
            time_as_float=time_as_float,
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
    target_drift, target_drift_kind = _target_drift(
        y_full, target_valid, target_kind, time_as_float,
    ) if y_full is not None else (None, None)

    return TemporalReport(
        time_column=time_col,
        target=target,
        signals=signals,
        target_drift=target_drift,
        target_drift_kind=target_drift_kind,
        insufficient=None,
    )


# --- per-feature analysis --------------------------------------------------

def _analyze_feature(
    feat: str,
    kind: str,
    x_raw: np.ndarray,
    feat_valid: np.ndarray,
    time_as_float: np.ndarray,
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

    split_point = int(early_idx[-1]) + 1 if len(early_idx) else 0
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
            and _enough_test_positives(y_full[test_idx], target_kind)
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
            and _enough_test_positives(y_full[rand_test], target_kind)
        ):
            random_pps = pps(
                x_enc.reshape(-1, 1), y_full, target_kind,
                split=(rand_train, rand_test),
            )

    severity, reason = _classify(
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
    )


def _classify(
    random_pps: float | None,
    time_pps: float | None,
    drift_ks: float | None,
    time_monotonicity: float | None,
    n_unique: int,
    n_nonnull: int,
) -> tuple[str, str]:
    # Leakage: predictive power collapses on time-ordered eval
    if (
        random_pps is not None and time_pps is not None
        and random_pps >= LEAK_MIN_RANDOM_PPS
        and (random_pps - time_pps) >= LEAK_GAP_THRESHOLD
    ):
        return "critical", (
            f"Predicts target on random CV ({random_pps:.2f}) but fails on "
            f"time-ordered split ({time_pps:.2f}). Likely contains future information."
        )

    # Drift + non-trivial signal
    if (
        drift_ks is not None and drift_ks >= DRIFT_KS_THRESHOLD
        and ((random_pps or 0) >= DRIFT_MIN_PPS or (time_pps or 0) >= DRIFT_MIN_PPS)
    ):
        return "warning", (
            f"Distribution shifts over time (KS={drift_ks:.2f}) on a predictive feature. "
            "Production model may degrade."
        )

    # Time-monotonic + unique-per-row
    if (
        time_monotonicity is not None and time_monotonicity >= MONOTONIC_THRESHOLD
        and n_unique >= n_nonnull * 0.9
    ):
        return "warning", (
            f"Strictly increases/decreases with time (Spearman={time_monotonicity:.2f}) "
            "and is unique-per-row — likely a row counter or ingest timestamp."
        )

    return "none", ""


def _enough_test_positives(y: np.ndarray, target_kind: str) -> bool:
    if target_kind != "classification":
        return True
    return (
        int((y == 1).sum()) >= MIN_TEST_POSITIVES_CLASSIF
        or int((y == 0).sum()) >= MIN_TEST_POSITIVES_CLASSIF
    )


# --- helpers ---------------------------------------------------------------

def _time_to_float(values: np.ndarray) -> np.ndarray:
    """Convert temporal values to a float scalar suitable for ranking."""
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None:
            out[i] = np.nan
            continue
        try:
            out[i] = float(np.datetime64(v).astype("datetime64[s]").astype(np.int64))
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
            stat = ks_2samp(early_f, late_f, alternative="two-sided", method="auto").statistic
            return float(stat)
        except Exception:
            return None
    # categorical / boolean: total variation distance on top-k frequencies
    early_str = early.astype(str)
    late_str = late.astype(str)
    cats = np.unique(np.concatenate([early_str, late_str]))
    if len(cats) > 100:
        return None
    e_freq = np.array([(early_str == c).mean() for c in cats])
    l_freq = np.array([(late_str == c).mean() for c in cats])
    return float(0.5 * np.abs(e_freq - l_freq).sum())


def _target_drift(
    y_full: np.ndarray | None,
    target_valid: np.ndarray | None,
    target_kind: str | None,
    time_as_float: np.ndarray,
) -> tuple[float | None, str | None]:
    """Drift across time deciles, with the kind label honest about its units.

    Returns:
      ("binary",            max_rate - min_rate)    — only for 2-class targets
      ("regression_ratio",  max_mean / min_mean)    — strictly-positive regression targets
      ("regression_diff",   max_mean - min_mean)    — regression targets that can be ≤ 0
      ("multiclass",        max_class_rate_drift)   — for K>2 classification: maximum
                                                       per-class rate range across deciles
    """
    if y_full is None or target_valid is None or target_kind is None:
        return None, None
    mask = target_valid & ~np.isnan(time_as_float)
    if mask.sum() < 100:
        return None, None
    y = y_full[mask]
    t = time_as_float[mask]
    order = np.argsort(t)
    y_sorted = y[order]

    # split rows into 10 chronological deciles
    deciles = [d for d in np.array_split(y_sorted, 10) if len(d) > 0]
    if len(deciles) < 5:
        return None, None

    if target_kind == "classification":
        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            # treat 1 as positive class regardless of label encoding
            pos = unique_classes.max()
            rates = np.array([(d == pos).mean() for d in deciles])
            return float(rates.max() - rates.min()), "binary"
        # multi-class: per-class positive-rate range across deciles; report max
        max_drift = 0.0
        for c in unique_classes:
            rates = np.array([(d == c).mean() for d in deciles])
            max_drift = max(max_drift, float(rates.max() - rates.min()))
        return max_drift, "multiclass"

    # regression
    means = np.array([float(d.astype(np.float64).mean()) for d in deciles])
    if means.min() > 0:
        return float(means.max() / means.min()), "regression_ratio"
    return float(means.max() - means.min()), "regression_diff"


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
    # regression_diff: scale-dependent, so we can't apply a universal threshold;
    # surface only when the difference is large in absolute terms relative to
    # the data — for v1, do not auto-flag (caller decides).
    return False
