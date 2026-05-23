"""Correlation analyses: Pearson (linear), mutual information (non-linear),
and target-aware predictive signal."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from biopsy.io import Source
from biopsy.matrix import SampleCache, _fetch_object_array
from biopsy.stats import ColumnStats, _quote


@dataclass
class CorrelationPair:
    a: str
    b: str
    pearson: float | None
    mutual_info: float | None  # normalized to [0, 1]

    @property
    def score(self) -> float:
        """Best available association strength in [0, 1]."""
        candidates = []
        if self.pearson is not None and not np.isnan(self.pearson):
            candidates.append(abs(self.pearson))
        if self.mutual_info is not None:
            candidates.append(self.mutual_info)
        return max(candidates) if candidates else 0.0

    @property
    def is_nonlinear(self) -> bool:
        """MI noticeably exceeds Pearson — relationship has a non-linear component."""
        if self.mutual_info is None or self.pearson is None:
            return False
        return self.mutual_info - abs(self.pearson) > 0.15


def pearson_matrix(
    src: Source,
    stats: dict[str, ColumnStats],
    *,
    max_cols: int | None = None,
    priority_features: list[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Pearson via DuckDB's corr() — fast, no row transfer.

    When `max_cols` is set, restricts the pairwise pass to the top-N numeric
    columns, preferring those listed in `priority_features`. This cuts the
    O(n²) corr() projection on wide datasets — at 500 numeric columns the
    uncapped pass builds ~125k aggregates per row scan.
    """
    numeric = [n for n, s in stats.items() if s.kind == "numeric" and not s.is_constant]
    if max_cols is not None and len(numeric) > max_cols:
        if priority_features:
            priority = [c for c in priority_features if c in numeric]
            rest = [c for c in numeric if c not in priority]
            numeric = (priority + rest)[:max_cols]
        else:
            numeric = numeric[:max_cols]
    pairs: dict[tuple[str, str], float] = {}
    if len(numeric) < 2:
        return pairs

    select_parts = []
    keys = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            select_parts.append(f"corr({_quote(a)}, {_quote(b)})")
            keys.append((a, b))
    if not select_parts:
        return pairs
    row = src.con.execute(f"SELECT {', '.join(select_parts)} FROM data").fetchone()
    for key, val in zip(keys, row, strict=True):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            pairs[key] = float(val)
    return pairs


def _valid_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of non-null entries in an object/numeric array."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return ~np.isnan(arr)
    if np.issubdtype(arr.dtype, np.datetime64):
        return ~np.isnat(arr)
    if arr.dtype != object:
        return np.ones(arr.shape, dtype=bool)
    return np.asarray([_is_valid_value(v) for v in arr], dtype=bool)


def _is_valid_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return not np.isnan(value)
    if isinstance(value, np.floating):
        return not bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return not bool(np.isnat(value))
    return True


def _encode(values: np.ndarray, kind: str) -> tuple[np.ndarray, bool]:
    """Encode a *clean* (no nulls) 1D array. Returns (array, is_discrete)."""
    if kind == "numeric":
        return np.asarray(values, dtype=np.float64), False
    encoded = LabelEncoder().fit_transform(np.asarray(values, dtype=str))
    return encoded.astype(np.float64), True


def mutual_info_matrix(
    src: Source,
    stats: dict[str, ColumnStats],
    max_rows: int = 20_000,
    sample_cache: SampleCache | None = None,
    *,
    max_cols: int | None = None,
    priority_features: list[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Pairwise mutual information for numeric + low-cardinality categorical columns.

    Normalized to [0, 1] via 1 - exp(-2 * MI), a standard transform that maps MI to a
    correlation-like scale.

    When `max_cols` is set, restricts the pairwise pass to the top-N columns,
    preferring those listed in `priority_features` (e.g., ranked by target
    signal). This cuts the O(n²) MI cost on wide datasets.
    """
    eligible = [
        n for n, s in stats.items()
        if not s.is_constant
        and (s.kind == "numeric" or (s.kind in {"text", "bool"} and s.n_unique <= 50))
    ]
    if max_cols is not None and len(eligible) > max_cols:
        if priority_features:
            priority = [c for c in priority_features if c in eligible]
            rest = [c for c in eligible if c not in priority]
            eligible = (priority + rest)[:max_cols]
        else:
            eligible = eligible[:max_cols]
    if len(eligible) < 2:
        return {}

    if sample_cache is None:
        quoted = ", ".join(_quote(c) for c in eligible)
        raw = _fetch_object_array(
            src.con,
            f"SELECT {quoted} FROM data USING SAMPLE {max_rows} ROWS (reservoir, 42)",
            eligible,
        )
        if raw.size == 0:
            return {}
    else:
        _cols, raw = sample_cache.fetch(eligible, max_rows=max_rows)
    if raw.size == 0:
        return {}
    masks = [_valid_mask(raw[:, j]) for j in range(len(eligible))]

    pair_args: list[tuple[str, str, int, int]] = [
        (eligible[i], eligible[j], i, j)
        for i in range(len(eligible))
        for j in range(i + 1, len(eligible))
    ]
    pairs: dict[tuple[str, str], float] = {}
    if not pair_args:
        return pairs

    def _mi_pair(a: str, b: str, i: int, j: int) -> tuple[str, str, float | None]:
        valid = masks[i] & masks[j]
        if valid.sum() < 30:
            return a, b, None
        ea, da = _encode(raw[valid, i], stats[a].kind)
        eb, db = _encode(raw[valid, j], stats[b].kind)
        try:
            if db:
                mi = mutual_info_classif(
                    ea.reshape(-1, 1), eb.astype(int),
                    discrete_features=[da], random_state=42,
                )[0]
            else:
                mi = mutual_info_regression(
                    ea.reshape(-1, 1), eb, discrete_features=[da], random_state=42,
                )[0]
        except ValueError:
            # sklearn raises ValueError for degenerate inputs (single sample
            # per class, all-NaN, etc.). Skip the pair rather than abort.
            return a, b, None
        return a, b, float(1 - np.exp(-2 * max(mi, 0)))

    # Parallelize when the pair count justifies the joblib fork overhead.
    # 64 pairs ≈ 12 columns; sklearn's MI on a single pair is fast enough that
    # smaller problems are net slower with multi-process dispatch.
    results: list[tuple[str, str, float | None]]
    if len(pair_args) >= 64:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(_mi_pair)(a, b, i, j) for a, b, i, j in pair_args
        )
    else:
        results = [_mi_pair(*args) for args in pair_args]

    for a, b, value in results:
        if value is not None:
            pairs[(a, b)] = value
    return pairs


def correlation_pairs(
    src: Source,
    stats: dict[str, ColumnStats],
    *,
    include_mutual_info: bool = True,
    sample_cache: SampleCache | None = None,
    max_cols: int | None = None,
    priority_features: list[str] | None = None,
) -> list[CorrelationPair]:
    pearson = pearson_matrix(
        src, stats, max_cols=max_cols, priority_features=priority_features,
    )
    mi = (
        mutual_info_matrix(
            src, stats,
            sample_cache=sample_cache,
            max_cols=max_cols,
            priority_features=priority_features,
        )
        if include_mutual_info else {}
    )

    keys = set(pearson) | set(mi)
    out = [
        CorrelationPair(a=a, b=b, pearson=pearson.get((a, b)), mutual_info=mi.get((a, b)))
        for (a, b) in keys
    ]
    out.sort(key=lambda p: p.score, reverse=True)
    return out


@dataclass
class TargetSignal:
    feature: str
    score: float                        # PPS-style out-of-sample predictive score, [0, 1]
    mutual_info: float                  # MI-normalized score, [0, 1]
    method: str                         # "pps_classif" | "pps_regress"
    spearman: float | None = None       # signed rank correlation, [-1, 1]
    auc: float | None = None            # AUC lift: normalized 2·|AUC−0.5|, binary only
    raw_auc: float | None = None        # raw ROC-AUC, binary classif only, [0, 1]
    perm_importance: float | None = None  # relative multivariate permutation importance
    support: int = 0
    positive_count: int | None = None
    is_leak_suspect: bool = False
    # CIs and stability — populated when the caller opts in via bootstrap /
    # multi-seed PPS. `None` means the metric wasn't recomputed.
    auc_ci_low: float | None = None
    auc_ci_high: float | None = None
    mi_ci_low: float | None = None
    mi_ci_high: float | None = None
    pps_stability: float | None = None  # coefficient of variation across seeds

    @property
    def confidence(self) -> str:
        """Qualitative confidence tier — `low` / `medium` / `high`.

        Tiers reflect effective sample size; binary classification also
        factors in positive-class support since rare-event metrics are
        unstable until the positive class has enough rows.
        """
        if self.positive_count is not None:
            if self.positive_count < 30 or self.support < 200:
                return "low"
            if self.positive_count < 100 or self.support < 1000:
                return "medium"
            return "high"
        if self.support < 200:
            return "low"
        if self.support < 1000:
            return "medium"
        return "high"

    @property
    def best_score(self) -> float:
        """Highest signal strength across robust metrics. Used for ranking when
        PPS is degenerate (extremely imbalanced targets).

        Spearman is intentionally excluded — it can saturate near 1.0 on
        sparse or near-constant inputs, overstating real signal. (Matches the
        criteria used by clustering._score_for_ranking.)
        """
        candidates = [self.score, self.mutual_info]
        if self.auc is not None:
            candidates.append(self.auc)
        if self.perm_importance is not None:
            candidates.append(self.perm_importance)
        return max(candidates)


def _pps_classification(
    X: np.ndarray,
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """F1 (weighted) of a depth-capped decision tree vs. majority-class baseline.

    split=None → cross-validated. split=(train_idx, test_idx) → single holdout.
    Normalized to [0, 1] where 0 = no better than naive, 1 = perfect.
    """
    if len(np.unique(y)) < 2:
        return 0.0
    # Determine safe CV fold count: needs at least 2 samples of the minority
    # class. If the minority has fewer than 2, cross-validated PPS is undefined.
    if y.min() >= 0:
        minority = int(np.bincount(y).min())
        if minority < 2:
            return 0.0
        cv_folds = max(2, min(4, minority))
    else:
        cv_folds = 4
    try:
        if split is None:
            scores = cross_val_score(
                DecisionTreeClassifier(max_depth=4, random_state=42),
                X, y, cv=cv_folds,
                scoring="f1_weighted",
            )
            model_score = float(scores.mean())
            baseline_y = y
            majority_source = y
        else:
            train_idx, test_idx = split
            if len(np.unique(y[train_idx])) < 2 or len(test_idx) < 10:
                return 0.0
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            model_score = f1_score(
                y[test_idx], preds, average="weighted", zero_division=0
            )
            baseline_y = y[test_idx]
            majority_source = y[train_idx]
    except ValueError:
        # sklearn raises ValueError on degenerate folds (e.g., a CV split with
        # only one class). Treat as no signal rather than failing the run.
        return 0.0
    majority = int(np.bincount(majority_source).argmax())
    naive_score = f1_score(
        baseline_y,
        np.full_like(baseline_y, majority),
        average="weighted",
        zero_division=0,
    )
    if model_score <= naive_score:
        return 0.0
    return float((model_score - naive_score) / (1 - naive_score + 1e-9))


def _pps_regression(
    X: np.ndarray,
    y: np.ndarray,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """MAE of a depth-capped tree vs. median baseline.

    split=None → cross-validated. split=(train_idx, test_idx) → single holdout.
    Normalized: PPS = max(0, 1 - model_mae / naive_mae).
    """
    try:
        if split is None:
            neg_mae = cross_val_score(
                DecisionTreeRegressor(max_depth=4, random_state=42),
                X, y, cv=4, scoring="neg_mean_absolute_error",
            )
            model_mae = -float(neg_mae.mean())
            baseline_y = y
            median_source = y
        else:
            train_idx, test_idx = split
            if len(train_idx) < 10 or len(test_idx) < 10:
                return 0.0
            reg = DecisionTreeRegressor(max_depth=4, random_state=42)
            reg.fit(X[train_idx], y[train_idx])
            preds = reg.predict(X[test_idx])
            model_mae = mean_absolute_error(y[test_idx], preds)
            baseline_y = y[test_idx]
            median_source = y[train_idx]
    except ValueError:
        # sklearn raises ValueError on degenerate inputs (all-NaN, empty fold).
        # Treat as no signal rather than failing the run.
        return 0.0
    naive_mae = mean_absolute_error(
        baseline_y,
        np.full_like(baseline_y, np.median(median_source)),
    )
    if naive_mae <= 0:
        return 0.0
    return float(max(0.0, 1 - model_mae / naive_mae))


def pps(
    X: np.ndarray,
    y: np.ndarray,
    target_kind: str,
    split: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Public PPS entry point — dispatches to classification or regression."""
    if target_kind == "classification":
        return _pps_classification(X, y.astype(int), split=split)
    return _pps_regression(X, y, split=split)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Signed rank correlation. Uses scipy's average-rank implementation so
    tied values get tied ranks (avoids spurious correlations on near-constant
    inputs)."""
    if len(x) < 5 or len(y) != len(x):
        return None
    with warnings.catch_warnings():
        # constant input is a valid "no signal" case; we return None below.
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(x, y, nan_policy="omit")
    rho = float(result.statistic)
    if np.isnan(rho):
        return None
    return rho


def _auc_scores(y: np.ndarray, score: np.ndarray) -> tuple[float, float] | None:
    """Return (raw_auc, auc_lift). AUC lift normalizes so 0 = no signal, 1 = perfect."""
    if len(np.unique(y)) != 2:
        return None
    raw = float(roc_auc_score(y, score))
    return raw, float(2 * abs(raw - 0.5))


def _score_feature_vs_target(
    feat: str,
    raw: np.ndarray,
    j: int,
    stats: dict[str, ColumnStats],
    y_full: np.ndarray,
    target_mask: np.ndarray,
    target_kind: str,
    is_binary_target: bool,
) -> tuple[TargetSignal, np.ndarray, bool, np.ndarray] | None:
    """Score one feature against the target; returns signal + permutation record."""
    feat_mask = _valid_mask(raw[:, j]) & target_mask
    if feat_mask.sum() < 30:
        return None

    x_enc, x_disc = _encode(raw[feat_mask, j], stats[feat].kind)
    y_sub_mask = feat_mask[target_mask]
    y_sub = y_full[y_sub_mask]
    X = x_enc.reshape(-1, 1)

    try:
        if target_kind == "classification":
            mi = mutual_info_classif(
                X, y_sub.astype(int), discrete_features=[x_disc], random_state=42
            )[0]
            pps_score = _pps_classification(X, y_sub.astype(int))
            method = "pps_classif"
        else:
            mi = mutual_info_regression(
                X, y_sub, discrete_features=[x_disc], random_state=42
            )[0]
            pps_score = _pps_regression(X, y_sub)
            method = "pps_regress"
    except ValueError:
        return None

    mi_norm = float(1 - np.exp(-2 * max(mi, 0)))

    spearman = None
    if stats[feat].kind == "numeric" and (
        target_kind == "regression" or is_binary_target
    ):
        spearman = _spearman(x_enc, y_sub.astype(np.float64))

    raw_auc = None
    auc = None
    if is_binary_target:
        auc_pair = _auc_scores(y_sub.astype(int), x_enc)
        if auc_pair is not None:
            raw_auc, auc = auc_pair

    minority_count = None
    if is_binary_target:
        counts = np.bincount(y_sub.astype(int), minlength=2)
        minority_count = int(counts.min())
    signal = TargetSignal(
        feature=feat,
        score=pps_score,
        mutual_info=mi_norm,
        method=method,
        spearman=spearman,
        auc=auc,
        raw_auc=raw_auc,
        support=int(feat_mask.sum()),
        positive_count=minority_count,
    )
    return signal, x_enc, x_disc, feat_mask


def target_signal(
    src: Source,
    stats: dict[str, ColumnStats],
    target: str,
    max_rows: int = 30_000,
    *,
    include_permutation: bool = True,
    stratify: bool = True,
    bootstrap: int = 0,
    pps_seeds: int = 1,
) -> list[TargetSignal]:
    """Rank features by association with the target.

    For numeric targets: MI regression + PPS regression. For low-cardinality
    targets: MI classification + PPS classification + (binary only) AUC. Adds
    Spearman for numeric features, plus multivariate permutation importance.
    """
    if target not in stats:
        raise ValueError(f"Target column not found: {target}")
    from biopsy.targets import target_task_kind

    t_stats = stats[target]
    target_kind = target_task_kind(t_stats)

    features = [
        n for n, s in stats.items()
        if n != target
        and not s.is_constant
        and (s.kind == "numeric" or (s.kind in {"text", "bool"} and s.n_unique <= 100))
    ]
    if not features:
        return []

    raw = _target_sample(
        src=src,
        cols=[*features, target],
        target=target,
        target_kind=target_kind,
        n_unique=t_stats.n_unique,
        max_rows=max_rows,
        stratify=stratify,
    )
    if raw.shape[0] < 30:
        return []
    target_mask = _valid_mask(raw[:, -1])

    if target_kind == "classification":
        y_full = LabelEncoder().fit_transform(raw[target_mask, -1].astype(str)).astype(int)
    else:
        y_full = np.asarray(raw[target_mask, -1], dtype=np.float64)

    is_binary_target = target_kind == "classification" and len(np.unique(y_full)) == 2

    # collect encoded per-feature arrays for the optional joint permutation importance
    feat_records: list[tuple[str, np.ndarray, bool, np.ndarray]] = []  # (name, x_enc, x_disc, mask)

    signals: list[TargetSignal] = []
    for j, feat in enumerate(features):
        scored = _score_feature_vs_target(
            feat, raw, j, stats, y_full, target_mask, target_kind, is_binary_target,
        )
        if scored is None:
            continue
        signal, x_enc, x_disc, feat_mask = scored
        signals.append(signal)
        feat_records.append((feat, x_enc, x_disc, feat_mask))

    if include_permutation:
        # Multivariate permutation importance from a single RF fit on the intersection
        # of rows where every feature is valid. Skipped if too few rows survive.
        _attach_permutation_importance(
            signals, feat_records, y_full, target_mask, target_kind,
        )

    # If PPS is degenerate across the board (e.g., extreme class imbalance makes
    # naive baseline unbeatable), fall back to best-available-metric ranking so
    # AUC / permutation importance can surface real signal.
    max_pps = max((s.score for s in signals), default=0.0)
    if max_pps < 0.05:
        signals.sort(key=lambda s: s.best_score, reverse=True)
    else:
        signals.sort(key=lambda s: s.score, reverse=True)

    if bootstrap > 0 or pps_seeds > 1:
        _attach_uncertainty(
            signals, feat_records, y_full, target_mask, target_kind, is_binary_target,
            n_bootstrap=bootstrap, n_pps_seeds=pps_seeds,
        )

    # leakage heuristic: PPS ≥ 0.85, OR PPS ≥ 0.6 AND >= 2× the next best feature.
    for i, s in enumerate(signals):
        if (
            s.score >= 0.85
            or (
                s.score >= 0.6
                and i + 1 < len(signals)
                and s.score >= 2 * signals[i + 1].score
            )
        ):
            s.is_leak_suspect = True
    return signals


def _target_sample(
    src: Source,
    cols: list[str],
    target: str,
    target_kind: str,
    n_unique: int,
    max_rows: int,
    stratify: bool,
) -> np.ndarray:
    """Pull target-analysis rows as a 2D object array (column order = cols).

    For low-cardinality classification targets, use deterministic per-class
    sampling so rare positives do not disappear from the analysis sample.
    """
    quoted = ", ".join(_quote(c) for c in cols)
    qtarget = _quote(target)
    if stratify and target_kind == "classification" and 1 < n_unique <= 20:
        per_class = max(1, max_rows // n_unique)
        sql = f"""
            WITH labeled AS (
                SELECT row_number() OVER () AS __biopsy_rowid, {quoted}
                FROM data
                WHERE {qtarget} IS NOT NULL
            ),
            ranked AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY {qtarget}
                           ORDER BY hash(__biopsy_rowid)
                       ) AS __biopsy_rank
                FROM labeled
            )
            SELECT {quoted}
            FROM ranked
            WHERE __biopsy_rank <= {int(per_class)}
        """
    else:
        sql = (
            f"SELECT {quoted} FROM ("
            f"SELECT {quoted} FROM data WHERE {qtarget} IS NOT NULL"
            f") USING SAMPLE {max_rows} ROWS (reservoir, 42)"
        )
    return _fetch_object_array(src.con, sql, cols)


def _attach_permutation_importance(
    signals: list[TargetSignal],
    feat_records: list[tuple[str, np.ndarray, bool, np.ndarray]],
    y_full: np.ndarray,
    target_mask: np.ndarray,
    target_kind: str,
) -> None:
    """Fit one RandomForest on all features, then compute permutation importance.

    Nulls are median-imputed per feature so wide tables with sparse columns still
    yield a meaningful joint design matrix.
    """
    if len(feat_records) < 2:
        return

    n = int(target_mask.sum())
    cols = []
    kept_signal_idx: list[int] = []  # parallel to cols; index into `signals`
    for i, (_name, x_enc, _disc, fmask) in enumerate(feat_records):
        feat_in_target = fmask[target_mask]
        col = np.full(n, np.nan, dtype=np.float64)
        col[feat_in_target] = x_enc
        if np.isnan(col).any():
            valid = ~np.isnan(col)
            if valid.sum() < 30:
                continue
            col[~valid] = float(np.median(col[valid]))
        cols.append(col)
        kept_signal_idx.append(i)

    if len(cols) < 2:
        return

    X = np.column_stack(cols)
    y = y_full
    if X.shape[0] < 100:
        return

    if target_kind == "classification":
        if len(np.unique(y)) < 2:
            return
        # class_weight='balanced' helps the model attend to rare classes;
        # AUC scoring is robust to class imbalance for binary targets.
        model = RandomForestClassifier(
            n_estimators=80, max_depth=6, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "f1_weighted"
    else:
        model = RandomForestRegressor(
            n_estimators=80, max_depth=6, random_state=42, n_jobs=-1,
        )
        scoring = "neg_mean_absolute_error"
    try:
        model.fit(X, y)
        result = permutation_importance(
            model, X, y, n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1,
        )
    except ValueError:
        # Degenerate inputs (e.g., scoring="roc_auc" on a fold that loses one
        # class after permutation). Skip permutation importance for this run;
        # PPS/MI/AUC have already been recorded per-feature.
        return

    importances = result.importances_mean
    if importances.max() <= 0:
        return
    normalized = importances / importances.max()
    for sig_idx, imp in zip(kept_signal_idx, normalized, strict=True):
        signals[sig_idx].perm_importance = float(max(imp, 0.0))


def _attach_uncertainty(
    signals: list[TargetSignal],
    feat_records: list[tuple[str, np.ndarray, bool, np.ndarray]],
    y_full: np.ndarray,
    target_mask: np.ndarray,
    target_kind: str,
    is_binary_target: bool,
    *,
    n_bootstrap: int,
    n_pps_seeds: int,
) -> None:
    """Compute 95% bootstrap CIs on AUC + MI and a multi-seed PPS stability
    score. Mutates `signals` in-place; opt-in via target_signal() args."""
    feats_by_name = {name: (x_enc, x_disc, fmask) for name, x_enc, x_disc, fmask in feat_records}
    rng_root = np.random.default_rng(42)

    for s in signals:
        record = feats_by_name.get(s.feature)
        if record is None:
            continue
        x_enc, x_disc, fmask = record
        y_sub_mask = fmask[target_mask]
        y_sub = y_full[y_sub_mask]
        n = len(y_sub)
        if n < 50:
            continue

        # --- bootstrap CIs on AUC + MI ---
        if n_bootstrap > 0:
            aucs: list[float] = []
            mis: list[float] = []
            for _ in range(n_bootstrap):
                idx = rng_root.integers(0, n, size=n)
                x_b = x_enc[idx]
                y_b = y_sub[idx]
                if is_binary_target and len(np.unique(y_b)) == 2:
                    pair = _auc_scores(y_b.astype(int), x_b)
                    if pair is not None:
                        aucs.append(pair[0])
                try:
                    if target_kind == "classification":
                        mi = mutual_info_classif(
                            x_b.reshape(-1, 1), y_b.astype(int),
                            discrete_features=[x_disc], random_state=42,
                        )[0]
                    else:
                        mi = mutual_info_regression(
                            x_b.reshape(-1, 1), y_b,
                            discrete_features=[x_disc], random_state=42,
                        )[0]
                    mis.append(float(1 - np.exp(-2 * max(mi, 0))))
                except ValueError:
                    pass
            if aucs:
                s.auc_ci_low = float(np.quantile(aucs, 0.025))
                s.auc_ci_high = float(np.quantile(aucs, 0.975))
            if mis:
                s.mi_ci_low = float(np.quantile(mis, 0.025))
                s.mi_ci_high = float(np.quantile(mis, 0.975))

        # --- multi-seed PPS stability ---
        # Bootstrap-resample (x, y) pairs together so sample composition
        # genuinely changes; joint-permutation of (x, y) by the same index
        # would just re-order pairs and PPS would be invariant.
        if n_pps_seeds > 1:
            scores: list[float] = []
            for seed in range(n_pps_seeds):
                rng_seed = np.random.default_rng(seed)
                idx = rng_seed.integers(0, n, size=n)
                x_b = x_enc[idx].reshape(-1, 1)
                y_b = y_sub[idx]
                try:
                    if target_kind == "classification":
                        score = _pps_classification(x_b, y_b.astype(int))
                    else:
                        score = _pps_regression(x_b, y_b)
                except ValueError:
                    continue
                scores.append(score)
            if len(scores) >= 2:
                arr = np.asarray(scores, dtype=float)
                mean = float(arr.mean())
                if mean > 0:
                    s.pps_stability = float(arr.std(ddof=1) / mean)
