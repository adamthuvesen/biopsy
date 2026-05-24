"""Profile-to-profile drift comparison.

`compare_profiles(a, b)` consumes two `Profile` objects and produces a
`CompareReport` that ranks per-column drift, surfaces schema diffs, and
quantifies target movement. Both `biopsy compare data1 data2` and
`biopsy compare a.json b.json` produce two `Profile`s and call this — the
core never touches raw data, so JSON round-tripped profiles work too.

Numeric drift uses histogram-derived approximations: a bin-unioned CDF for
KS, signed bin probabilities for Wasserstein (1D) and PSI. Categorical drift
uses top-value counts to build a contingency table for chi-square and a
top-K probability distribution for Jensen-Shannon divergence.

Approximations matter: bin counts on the same column may have come from
different sample sizes or bin edges. We rebin to the union of edges before
comparing, and report `None` when neither side has the structure required
for a given metric (e.g., a categorical column has no histogram).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, kstwo, wasserstein_distance

from biopsy.findings import Finding
from biopsy.profile import Profile
from biopsy.stats import _NUMERIC_TYPES, ColumnStats

# --- thresholds ------------------------------------------------------------
KS_CRITICAL = 0.20
KS_WARNING = 0.10
PSI_CRITICAL = 0.25
PSI_WARNING = 0.10
JS_CRITICAL = 0.25
JS_WARNING = 0.10
NULL_DELTA_WARNING = 0.05


@dataclass
class SchemaDiff:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    type_changed: list[tuple[str, str, str]] = field(default_factory=list)
    shared: list[str] = field(default_factory=list)


@dataclass
class FeatureDrift:
    column: str
    kind: str  # "numeric" | "categorical" | "temporal" | "other"
    ks_stat: float | None = None
    ks_pvalue: float | None = None
    wasserstein: float | None = None
    psi: float | None = None
    chi2_pvalue: float | None = None
    js_divergence: float | None = None
    null_rate_a: float | None = None
    null_rate_b: float | None = None
    null_rate_delta: float | None = None
    mean_a: float | None = None
    mean_b: float | None = None
    mean_delta: float | None = None

    @property
    def drift_score(self) -> float:
        """Combined indicator in [0, 1] — max of normalized metric weights."""
        candidates: list[float] = []
        if self.ks_stat is not None:
            candidates.append(min(1.0, self.ks_stat / 0.5))
        if self.psi is not None:
            candidates.append(min(1.0, self.psi / 0.5))
        if self.js_divergence is not None:
            candidates.append(min(1.0, self.js_divergence / 0.5))
        if self.null_rate_delta is not None:
            candidates.append(min(1.0, abs(self.null_rate_delta) / 0.25))
        return max(candidates) if candidates else 0.0


@dataclass
class TargetDelta:
    kind: str  # "binary_rate" | "multiclass_rate" | "regression_mean" | "missing"
    a_value: float | None
    b_value: float | None
    delta: float | None
    detail: str


@dataclass
class CompareReport:
    a_name: str
    b_name: str
    schema: SchemaDiff
    drifts: list[FeatureDrift]  # sorted by drift_score desc
    target: TargetDelta | None
    findings: list[Finding]

    def top(self, limit: int = 10) -> list[FeatureDrift]:
        return self.drifts[:limit]


# ---------------------------------------------------------------------------


def compare_profiles(a: Profile, b: Profile) -> CompareReport:
    schema = _schema_diff(a, b)
    drifts: list[FeatureDrift] = []
    for col in schema.shared:
        sa = a.columns[col]
        sb = b.columns[col]
        if sa.kind == "numeric" and sb.kind == "numeric":
            drifts.append(_numeric_drift(col, sa, sb))
        elif sa.kind in {"text", "bool"} and sb.kind in {"text", "bool"}:
            drifts.append(_categorical_drift(col, sa, sb))
        else:
            drifts.append(_other_drift(col, sa, sb))
    drifts.sort(key=lambda d: d.drift_score, reverse=True)

    target = _target_delta(a, b)
    findings = _drift_findings(schema, drifts, target)
    return CompareReport(
        a_name=a.source_name,
        b_name=b.source_name,
        schema=schema,
        drifts=drifts,
        target=target,
        findings=findings,
    )


# --- schema -----------------------------------------------------------------


def _schema_diff(a: Profile, b: Profile) -> SchemaDiff:
    a_cols = set(a.columns)
    b_cols = set(b.columns)
    added = sorted(b_cols - a_cols)
    removed = sorted(a_cols - b_cols)
    shared_set = a_cols & b_cols
    shared = sorted(shared_set)
    type_changed: list[tuple[str, str, str]] = []
    for c in shared:
        if a.columns[c].kind != b.columns[c].kind:
            type_changed.append((c, a.columns[c].kind, b.columns[c].kind))
    return SchemaDiff(
        added=added,
        removed=removed,
        type_changed=type_changed,
        shared=shared,
    )


# --- numeric ---------------------------------------------------------------


def _numeric_drift(col: str, sa: ColumnStats, sb: ColumnStats) -> FeatureDrift:
    drift = FeatureDrift(column=col, kind="numeric")
    drift.null_rate_a = sa.null_rate
    drift.null_rate_b = sb.null_rate
    drift.null_rate_delta = sb.null_rate - sa.null_rate
    drift.mean_a = sa.mean
    drift.mean_b = sb.mean
    if sa.mean is not None and sb.mean is not None:
        drift.mean_delta = sb.mean - sa.mean

    rebinned = _rebin_to_union(sa, sb)
    if rebinned is None:
        return drift
    edges, counts_a, counts_b = rebinned
    n_a = int(counts_a.sum())
    n_b = int(counts_b.sum())
    if n_a < 30 or n_b < 30:
        return drift

    centers = 0.5 * (edges[:-1] + edges[1:])
    p_a = counts_a / n_a
    p_b = counts_b / n_b

    # KS on bin-derived CDFs — the statistic is the supremum of |F_a - F_b|
    # across the bin boundaries. The p-value uses the asymptotic Kolmogorov
    # distribution with the *original* sample sizes, not the synthetic
    # rehydrated ones the old path produced.
    cdf_a = np.cumsum(p_a)
    cdf_b = np.cumsum(p_b)
    ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))
    drift.ks_stat = ks_stat
    n_eff = (n_a * n_b) / (n_a + n_b)
    if n_eff > 0:
        drift.ks_pvalue = float(kstwo.sf(ks_stat, max(int(round(n_eff)), 1)))

    drift.wasserstein = float(wasserstein_distance(centers, centers, p_a, p_b))
    drift.psi = _psi_from_probs(p_a, p_b)
    return drift


def _rebin_to_union(
    sa: ColumnStats,
    sb: ColumnStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rebin both histograms onto a common edge grid spanning the union range.

    Returns (edges, counts_a, counts_b). Each side's mass is redistributed
    proportionally across the overlap between its source bins and the
    target bins, so total counts are preserved.
    """
    if not sa.histogram or not sb.histogram:
        return None
    lo = min(sa.histogram[0][0], sb.histogram[0][0])
    hi = max(sa.histogram[-1][1], sb.histogram[-1][1])
    if not (lo < hi):
        return None
    n_bins = max(len(sa.histogram), len(sb.histogram))
    edges = np.linspace(lo, hi, n_bins + 1)
    return edges, _rebin(sa.histogram, edges), _rebin(sb.histogram, edges)


def _rebin(
    histogram: list[tuple[float, float, int]],
    edges: np.ndarray,
) -> np.ndarray:
    """Redistribute histogram bin counts onto `edges` (preserving total count)."""
    out = np.zeros(len(edges) - 1, dtype=np.float64)
    for src_lo, src_hi, count in histogram:
        if count <= 0 or not (src_hi > src_lo):
            continue
        width = src_hi - src_lo
        # find overlap with each target bin
        i_lo = int(np.searchsorted(edges, src_lo, side="right") - 1)
        i_hi = int(np.searchsorted(edges, src_hi, side="left"))
        i_lo = max(0, min(i_lo, len(out) - 1))
        i_hi = max(i_lo, min(i_hi, len(out)))
        for i in range(i_lo, i_hi + 1):
            if i >= len(out):
                break
            ov_lo = max(src_lo, edges[i])
            ov_hi = min(src_hi, edges[i + 1])
            if ov_hi <= ov_lo:
                continue
            out[i] += count * (ov_hi - ov_lo) / width
    return out


def _psi_from_probs(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Population Stability Index on aligned bin probabilities."""
    # Laplace smoothing matches the original codepath; prevents log(0) when
    # one side has an empty bin the other side populates.
    n_bins = len(p_a)
    if n_bins < 3:
        return 0.0
    eps = 1.0 / max(n_bins, 1)
    a = (p_a + eps) / (1.0 + n_bins * eps)
    b = (p_b + eps) / (1.0 + n_bins * eps)
    return float(np.sum((b - a) * np.log(b / a)))


# --- categorical ----------------------------------------------------------


def _categorical_drift(col: str, sa: ColumnStats, sb: ColumnStats) -> FeatureDrift:
    drift = FeatureDrift(column=col, kind="categorical")
    drift.null_rate_a = sa.null_rate
    drift.null_rate_b = sb.null_rate
    drift.null_rate_delta = sb.null_rate - sa.null_rate

    # On near-unique categoricals (IDs, free text), top-K counts are noisy
    # by construction — each dataset's most-frequent values are essentially
    # random. Reporting "drift" on those is misleading.
    a_nonnull = max(sa.n - sa.n_null, 1)
    b_nonnull = max(sb.n - sb.n_null, 1)
    if (sa.n_unique / a_nonnull) > 0.5 or (sb.n_unique / b_nonnull) > 0.5:
        return drift

    keys = set()
    a_counts = dict(_normalize_top(sa.top_values))
    b_counts = dict(_normalize_top(sb.top_values))
    # Top-K must cover a meaningful share of the data for the K-bin JS
    # divergence and chi-square to be trustworthy. Below ~25% coverage,
    # we're just measuring which 10 random values happened to be most
    # frequent in each sample.
    a_top_coverage = sum(a_counts.values()) / a_nonnull
    b_top_coverage = sum(b_counts.values()) / b_nonnull
    if min(a_top_coverage, b_top_coverage) < 0.25:
        return drift
    keys.update(a_counts)
    keys.update(b_counts)
    if not keys:
        return drift
    a_vec = np.array([a_counts.get(k, 0) for k in keys], dtype=float)
    b_vec = np.array([b_counts.get(k, 0) for k in keys], dtype=float)
    if a_vec.sum() == 0 or b_vec.sum() == 0:
        return drift
    contingency = np.vstack([a_vec, b_vec])
    contingency = contingency[:, contingency.sum(axis=0) > 0]
    if contingency.shape[1] >= 2:
        # chi2_contingency raises on degenerate tables; treat as no signal.
        try:
            _chi2, p_value, _dof, _expected = chi2_contingency(contingency)
            drift.chi2_pvalue = float(p_value)
        except ValueError:
            drift.chi2_pvalue = None
    p = a_vec / a_vec.sum()
    q = b_vec / b_vec.sum()
    # scipy.spatial.distance.jensenshannon returns the JS *distance* (sqrt of
    # divergence); square it to recover divergence with natural-log base.
    js_distance = float(jensenshannon(p, q, base=math.e))
    drift.js_divergence = js_distance * js_distance
    return drift


def _normalize_top(items: list[tuple[object, int]]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for k, c in items:
        if not isinstance(c, _NUMERIC_TYPES):
            continue
        out.append((str(k), int(c)))
    return out


# --- other (temporal / unknown) -------------------------------------------


def _other_drift(col: str, sa: ColumnStats, sb: ColumnStats) -> FeatureDrift:
    drift = FeatureDrift(column=col, kind=sa.kind if sa.kind == sb.kind else "other")
    drift.null_rate_a = sa.null_rate
    drift.null_rate_b = sb.null_rate
    drift.null_rate_delta = sb.null_rate - sa.null_rate
    return drift


# --- target ----------------------------------------------------------------


def _target_delta(a: Profile, b: Profile) -> TargetDelta | None:
    if a.target is None or b.target is None or a.target != b.target:
        return None
    sa = a.target_summary
    sb = b.target_summary
    if sa is None or sb is None:
        return TargetDelta(
            kind="missing",
            a_value=None,
            b_value=None,
            delta=None,
            detail="target summary missing on one side",
        )

    if sa.kind == "classification" and sb.kind == "classification":
        if sa.positive_rate is not None and sb.positive_rate is not None:
            delta = sb.positive_rate - sa.positive_rate
            return TargetDelta(
                kind="binary_rate",
                a_value=sa.positive_rate,
                b_value=sb.positive_rate,
                delta=delta,
                detail=(
                    f"positive rate {sa.positive_rate:.2%} → {sb.positive_rate:.2%} "
                    f"(Δ {delta:+.2%})"
                ),
            )
        # multiclass: max per-class rate drift
        a_counts = dict(sa.class_counts)
        b_counts = dict(sb.class_counts)
        max_delta = 0.0
        for key in set(a_counts) | set(b_counts):
            a_rate = a_counts.get(key, 0) / max(sa.n - sa.n_null, 1)
            b_rate = b_counts.get(key, 0) / max(sb.n - sb.n_null, 1)
            max_delta = max(max_delta, abs(b_rate - a_rate))
        return TargetDelta(
            kind="multiclass_rate",
            a_value=None,
            b_value=None,
            delta=max_delta,
            detail=f"max per-class rate Δ = {max_delta:.2%}",
        )

    # regression mean (use the column stat from each profile)
    ca = a.columns.get(a.target)
    cb = b.columns.get(b.target)
    if ca is not None and cb is not None and ca.mean is not None and cb.mean is not None:
        delta = cb.mean - ca.mean
        return TargetDelta(
            kind="regression_mean",
            a_value=ca.mean,
            b_value=cb.mean,
            delta=delta,
            detail=f"mean {ca.mean:.4g} → {cb.mean:.4g} (Δ {delta:+.4g})",
        )
    return None


# --- findings --------------------------------------------------------------


def _drift_findings(
    schema: SchemaDiff,
    drifts: list[FeatureDrift],
    target: TargetDelta | None,
) -> list[Finding]:
    out: list[Finding] = []
    if schema.added:
        out.append(
            Finding(
                severity="info",
                category="drift",
                title=f"{len(schema.added)} new column(s) appear in B",
                detail=", ".join(schema.added[:10]),
                columns=list(schema.added),
                score=0.6,
            )
        )
    if schema.removed:
        out.append(
            Finding(
                severity="warning",
                category="drift",
                title=f"{len(schema.removed)} column(s) removed in B",
                detail=", ".join(schema.removed[:10]),
                columns=list(schema.removed),
                score=0.7,
            )
        )
    for col, ka, kb in schema.type_changed:
        out.append(
            Finding(
                severity="warning",
                category="drift",
                title=f"`{col}` changed type ({ka} → {kb})",
                detail="Dtype change between A and B — pipelines may break.",
                columns=[col],
                score=0.75,
            )
        )

    for d in drifts:
        sev = _drift_severity(d)
        if sev == "none":
            continue
        bits: list[str] = []
        if d.ks_stat is not None:
            valid_p = d.ks_pvalue is not None and not math.isnan(d.ks_pvalue)
            p_disp = f"{d.ks_pvalue:.2g}" if valid_p else "—"
            bits.append(f"KS={d.ks_stat:.2f} (p={p_disp})")
        if d.psi is not None and d.psi >= PSI_WARNING:
            bits.append(f"PSI={d.psi:.2f}")
        if d.js_divergence is not None and d.js_divergence >= JS_WARNING:
            bits.append(f"JS={d.js_divergence:.2f}")
        if d.chi2_pvalue is not None and d.chi2_pvalue < 0.05:
            bits.append(f"χ² p={d.chi2_pvalue:.2g}")
        if d.null_rate_delta is not None and abs(d.null_rate_delta) >= NULL_DELTA_WARNING:
            bits.append(f"null Δ {d.null_rate_delta:+.0%}")
        if not bits:
            continue
        out.append(
            Finding(
                severity=sev,
                category="drift",
                title=f"`{d.column}` drifts A → B",
                detail="; ".join(bits),
                columns=[d.column],
                score=d.drift_score,
            )
        )

    if target is not None and target.delta is not None:
        sev = "info"
        if target.kind == "binary_rate" and target.a_value:
            rel = abs(target.delta) / max(target.a_value, 1e-6)
            if rel >= 0.5:
                sev = "critical"
            elif rel >= 0.2:
                sev = "warning"
        elif target.kind == "multiclass_rate" and target.delta >= 0.05:
            sev = "warning" if target.delta < 0.15 else "critical"
        elif target.kind == "regression_mean":
            sev = "info"
        out.append(
            Finding(
                severity=sev,
                category="drift",
                title="target distribution shifts",
                detail=target.detail,
                columns=[],
                score=0.85 if sev == "critical" else 0.65,
            )
        )

    out.sort(key=lambda f: (f.rank, -f.score))
    return out


def _drift_severity(d: FeatureDrift) -> str:
    if d.ks_stat is not None and d.ks_stat >= KS_CRITICAL:
        return "critical"
    if d.psi is not None and d.psi >= PSI_CRITICAL:
        return "critical"
    if d.js_divergence is not None and d.js_divergence >= JS_CRITICAL:
        return "critical"
    if d.ks_stat is not None and d.ks_stat >= KS_WARNING:
        return "warning"
    if d.psi is not None and d.psi >= PSI_WARNING:
        return "warning"
    if d.js_divergence is not None and d.js_divergence >= JS_WARNING:
        return "warning"
    if d.null_rate_delta is not None and abs(d.null_rate_delta) >= NULL_DELTA_WARNING:
        return "warning"
    if d.chi2_pvalue is not None and d.chi2_pvalue < 0.05:
        return "info"
    return "none"
