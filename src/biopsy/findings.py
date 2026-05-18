"""Ranked findings — the opinionated 'what should I look at first' list."""

from __future__ import annotations

import re as _re
from dataclasses import dataclass
from typing import Any

from biopsy.correlations import CorrelationPair, TargetSignal
from biopsy.stats import ColumnStats
from biopsy.temporal import TemporalReport, is_target_drifted


@dataclass
class Finding:
    severity: str  # "critical" | "warning" | "info"
    category: str
    title: str
    detail: str
    columns: list[str]
    score: float  # for sorting within severity

    @property
    def rank(self) -> int:
        return {"critical": 0, "warning": 1, "info": 2}[self.severity]

    @property
    def why(self) -> str:
        if self.category == "leakage":
            return (
                "Leakage can make offline scores look excellent while production "
                "performance collapses."
            )
        if self.category == "quality":
            return (
                "Data quality issues change what the model can learn and often need "
                "a policy before training."
            )
        if self.category == "suspicious":
            return (
                "Suspicious columns are common sources of wasted features, accidental "
                "identifiers, or brittle encodings."
            )
        if self.category == "distribution":
            return (
                "Distribution shape affects transforms, model choice, and how much "
                "trust to put in summary statistics."
            )
        if self.category == "correlation":
            return (
                "Highly related features can duplicate signal, hide leakage, or make "
                "feature importance harder to interpret."
            )
        if self.category == "target":
            return (
                "Target issues determine whether metrics are stable and whether the "
                "problem framing is ready for modeling."
            )
        if self.category == "temporal":
            return (
                "Temporal effects reveal train/test mismatch and future-looking "
                "features before they become expensive modeling bugs."
            )
        return "This changes how the dataset should be prepared or interpreted."


_NULL_SENTINELS: frozenset[str] = frozenset({"?", "NA", "N/A", "n/a", "nan", "NaN"})

# ISO-8601-ish: YYYY-MM-DD or YYYY-MM-DDTHH:MM(:SS)?(Z|±HH:MM)?
# Anchored at the start; trailing text (e.g. " event", " UTC") is tolerated
# because DuckDB only keeps the column as VARCHAR when something prevents
# auto-cast — that "something" is usually a suffix.
_DATE_PATTERNS = [
    _re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?"),
    _re.compile(r"^\d{4}/\d{2}/\d{2}"),
]


def _looks_like_date_string(top_values: list[tuple[Any, int]]) -> bool:
    """At least 80% of the top sampled distinct values match an ISO date."""
    if not top_values:
        return False
    matched = 0
    total = 0
    for v, _c in top_values[:10]:
        if v is None:
            continue
        s = str(v).strip()
        total += 1
        for pat in _DATE_PATTERNS:
            if pat.match(s):
                matched += 1
                break
    return total > 0 and (matched / total) >= 0.8


def _values_are_bool_like(stats: ColumnStats) -> bool:
    """All numeric top values are exactly 0 or 1 (as numeric or string)."""
    if not stats.top_values:
        # If we don't have top_values, fall back to min/max bracket.
        if stats.min is not None and stats.max is not None:
            return float(stats.min) >= 0.0 and float(stats.max) <= 1.0 and stats.n_unique <= 2
        return False
    for v, _c in stats.top_values:
        try:
            n = float(str(v))
        except (TypeError, ValueError):
            return False
        if n not in (0.0, 1.0):
            return False
    return True


def _looks_like_id(name: str) -> bool:
    """Heuristic: column name signals an identifier.

    Match structural patterns only — `_id` suffix, exact `id` / `ID`, or
    `uuid` substring. Avoid bare `endswith("id")` because words like `paid`,
    `liquid`, `grid`, `valid` end with "id" without being IDs.
    """
    n = name.lower()
    if n == "id":
        return True
    if n.endswith("_id"):
        return True
    return "uuid" in n


def column_findings(
    stats: dict[str, ColumnStats],
    n_rows: int,
    target: str | None = None,
) -> list[Finding]:
    out: list[Finding] = []

    for s in stats.values():
        is_target = s.name == target
        # 100% null — check before constant, since an all-null column has
        # n_unique=0 and would otherwise be miscategorized as "constant".
        if s.null_rate >= 1.0:
            out.append(Finding(
                severity="critical", category="quality",
                title=f"`{s.name}` is 100% null",
                detail=f"All {s.n:,} rows are missing — column is empty.",
                columns=[s.name], score=1.0,
            ))
            continue

        # constants (after the all-null check above)
        if s.is_constant and not is_target:
            out.append(Finding(
                severity="warning", category="suspicious",
                title=f"`{s.name}` is constant",
                detail=(
                    f"Only {s.n_unique} unique value(s) across {n_rows:,} rows — "
                    "drop or investigate."
                ),
                columns=[s.name], score=1.0,
            ))
            continue

        # near-constant
        if s.is_near_constant and not is_target:
            top_val, top_count = s.top_values[0]
            pct = top_count / max(s.n - s.n_null, 1)
            out.append(Finding(
                severity="warning", category="suspicious",
                title=f"`{s.name}` is near-constant ({pct:.1%})",
                detail=(
                    f"Value '{top_val}' dominates {top_count:,} of "
                    f"{s.n - s.n_null:,} non-null rows."
                ),
                columns=[s.name], score=pct,
            ))

        # very high null rate
        if s.null_rate > 0.5:
            sev = "critical" if s.null_rate > 0.9 else "warning"
            out.append(Finding(
                severity=sev, category="quality",
                title=f"`{s.name}` is {s.null_rate:.0%} null",
                detail=f"{s.n_null:,} of {s.n:,} rows are missing.",
                columns=[s.name], score=s.null_rate,
            ))
        elif s.null_rate > 0.1:
            out.append(Finding(
                severity="info", category="quality",
                title=f"`{s.name}` has {s.null_rate:.0%} nulls",
                detail=f"{s.n_null:,} missing values.",
                columns=[s.name], score=s.null_rate,
            ))

        # ID-shaped feature (high cardinality + ID-ish name)
        if s.n_unique == s.n - s.n_null and s.n_unique > 50 and _looks_like_id(s.name):
            out.append(Finding(
                severity="warning", category="suspicious",
                title=f"`{s.name}` looks like an identifier",
                detail=(
                    f"All {s.n_unique:,} non-null values are unique — "
                    "unlikely to be predictive."
                ),
                columns=[s.name], score=0.9,
            ))

        # extreme skew
        if s.skew is not None and abs(s.skew) > 3 and not is_target:
            out.append(Finding(
                severity="info", category="distribution",
                title=f"`{s.name}` is heavily skewed (skew={s.skew:.1f})",
                detail="Consider log/Box-Cox transform before modeling.",
                columns=[s.name], score=min(abs(s.skew) / 10, 1.0),
            ))

        # outliers
        if s.n_outliers_iqr is not None and s.n_outliers_iqr > 0 and not is_target:
            rate = s.n_outliers_iqr / max(s.n - s.n_null, 1)
            if rate > 0.01:
                out.append(Finding(
                    severity="info", category="distribution",
                    title=f"`{s.name}` has {s.n_outliers_iqr:,} IQR outliers ({rate:.1%})",
                    detail=(
                        "Values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. "
                        f"Min={s.min:.4g}, max={s.max:.4g}."
                    ),
                    columns=[s.name], score=rate,
                ))

        # null sentinel encoded as a string value (e.g., "?" in CSV exports, "NA" from R)
        if s.kind == "text" and s.top_values:
            sentinel = next((v for v, _c in s.top_values if str(v) in _NULL_SENTINELS), None)
            if sentinel is not None:
                top_val, top_count = s.top_values[0]
                is_dominant = str(top_val) in _NULL_SENTINELS
                sev = "warning" if is_dominant else "info"
                out.append(Finding(
                    severity=sev, category="quality",
                    title=f"`{s.name}` contains encoded nulls ('{sentinel}')",
                    detail=(
                        f"'{sentinel}' appears as a value — a common null sentinel. "
                        "Replace with actual NULLs before profiling for correct statistics."
                    ),
                    columns=[s.name], score=0.85 if is_dominant else 0.6,
                ))

        # high-cardinality text (modeling foot-gun)
        if s.kind == "text" and s.n_unique > 0.5 * (s.n - s.n_null) and s.n_unique > 50:
            out.append(Finding(
                severity="info", category="suspicious",
                title=f"`{s.name}` has very high cardinality ({s.n_unique:,} unique)",
                detail="Free-text or near-unique — needs encoding or feature engineering.",
                columns=[s.name], score=0.7,
            ))

        # free-text columns (long average length + near-unique values)
        nonnull = s.n - s.n_null
        if (
            s.kind == "text"
            and s.avg_len is not None
            and s.avg_len > 32
            and nonnull > 0
            and (s.n_unique / nonnull) > 0.8
            and not is_target
        ):
            out.append(Finding(
                severity="info", category="suspicious",
                title=f"`{s.name}` looks like free text (avg_len={s.avg_len:.0f})",
                detail=(
                    "Long, near-unique strings — drop, hash, or tokenize. Naive "
                    "one-hot encoding will explode dimensionality."
                ),
                columns=[s.name], score=0.6,
            ))

        # date-string columns: text values that look like dates
        if (
            s.kind == "text"
            and not is_target
            and s.top_values
            and _looks_like_date_string(s.top_values)
        ):
            out.append(Finding(
                severity="warning", category="quality",
                title=f"`{s.name}` is a date stored as a string",
                detail=(
                    "Top values match an ISO-8601-ish date pattern. Cast to "
                    "DATE/TIMESTAMP before profiling so temporal checks fire."
                ),
                columns=[s.name], score=0.75,
            ))

        # bool-like integer columns: distinct values ⊆ {0, 1}
        if (
            s.kind == "numeric"
            and not is_target
            and s.n_unique <= 2
            and s.n_unique >= 1
            and _values_are_bool_like(s)
        ):
            out.append(Finding(
                severity="info", category="quality",
                title=f"`{s.name}` is a boolean stored as int",
                detail=(
                    "Only 0/1 values appear. Treat as a flag — no scaling, "
                    "mode imputation, no skew transform."
                ),
                columns=[s.name], score=0.4,
            ))

        # high-cardinality categorical that may end up target-encoded
        if (
            s.kind in {"text", "bool"}
            and not is_target
            and target is not None
            and nonnull > 0
            and s.n_unique > 0.3 * nonnull
            and s.n_unique > 30
        ):
            out.append(Finding(
                severity="warning", category="quality",
                title=f"`{s.name}` is high-cardinality — target encoding leakage risk",
                detail=(
                    f"{s.n_unique:,} unique levels over {nonnull:,} non-null rows. "
                    "If you target-encode this, fit the encoder out-of-fold."
                ),
                columns=[s.name], score=0.6,
            ))

    return out


def target_summary_findings(summary: Any) -> list[Finding]:
    out: list[Finding] = []
    if summary.kind == "classification":
        if summary.n_unique == 2 and summary.positive_count is not None:
            rate = summary.positive_rate or 0.0
            sev = "warning" if summary.positive_count < 100 or rate < 0.01 else "info"
            out.append(Finding(
                severity=sev,
                category="target",
                title=f"`{summary.name}` is an imbalanced binary target ({rate:.2%} positive)",
                detail=(
                    f"Positive class `{summary.positive_value}` appears "
                    f"{summary.positive_count:,} times across "
                    f"{summary.n - summary.n_null:,} labeled rows."
                ),
                columns=[summary.name],
                score=min(1.0, 1.0 - rate),
            ))
        elif summary.min_class_count is not None and summary.min_class_count < 30:
            out.append(Finding(
                severity="warning",
                category="target",
                title=f"`{summary.name}` has classes with very low support",
                detail=(
                    f"Smallest class has {summary.min_class_count:,} rows. "
                    "Target metrics may be unstable."
                ),
                columns=[summary.name],
                score=0.8,
            ))
    if summary.n_null:
        out.append(Finding(
            severity="warning" if summary.n_null / max(summary.n, 1) > 0.1 else "info",
            category="target",
            title=f"`{summary.name}` has missing target labels",
            detail=f"{summary.n_null:,} of {summary.n:,} rows have no target.",
            columns=[summary.name],
            score=summary.n_null / max(summary.n, 1),
        ))
    return out


def correlation_findings(pairs: list[CorrelationPair]) -> list[Finding]:
    out: list[Finding] = []
    for p in pairs[:15]:
        if p.score < 0.7:
            break
        # very high redundancy
        if p.pearson is not None and abs(p.pearson) > 0.95:
            out.append(Finding(
                severity="warning", category="correlation",
                title=f"`{p.a}` and `{p.b}` are nearly identical (r={p.pearson:+.3f})",
                detail="Strong multicollinearity — one likely derives from the other.",
                columns=[p.a, p.b], score=abs(p.pearson),
            ))
        elif p.is_nonlinear:
            out.append(Finding(
                severity="info", category="correlation",
                title=(
                    f"`{p.a}` ↔ `{p.b}` is non-linear "
                    f"(MI={p.mutual_info:.2f}, r={p.pearson:+.2f})"
                ),
                detail="Mutual information far exceeds linear correlation.",
                columns=[p.a, p.b], score=p.mutual_info or 0.0,
            ))
        elif p.score >= 0.8:
            r = f"r={p.pearson:+.2f}" if p.pearson is not None else f"MI={p.mutual_info:.2f}"
            out.append(Finding(
                severity="info", category="correlation",
                title=f"`{p.a}` ↔ `{p.b}` strongly associated ({r})",
                detail="",
                columns=[p.a, p.b], score=p.score,
            ))
    return out


def target_findings(signals: list[TargetSignal], target: str) -> list[Finding]:
    out: list[Finding] = []
    for s in signals[:10]:
        if s.is_leak_suspect:
            out.append(Finding(
                severity="critical", category="leakage",
                title=f"`{s.feature}` may leak the target (score={s.score:.2f})",
                detail=(
                    f"Predictive score against `{target}` is suspiciously high. "
                    "Check whether this column was computed from the target."
                ),
                columns=[s.feature, target], score=s.score,
            ))
        elif s.score >= 0.3:
            out.append(Finding(
                severity="info", category="target",
                title=f"`{s.feature}` → `{target}` (score={s.score:.2f})",
                detail="Notable predictive signal.",
                columns=[s.feature, target], score=s.score,
            ))
        if s.pps_stability is not None and s.pps_stability > 0.30 and s.score >= 0.05:
            out.append(Finding(
                severity="info", category="target",
                title=f"`{s.feature}` PPS is unstable (CoV={s.pps_stability:.2f})",
                detail=(
                    "Multi-seed PPS varies a lot for this feature — the ranking "
                    "may not be reliable. Consider holding out a larger sample."
                ),
                columns=[s.feature, target], score=0.4,
            ))
    return out


def temporal_findings(report: TemporalReport | None, target: str | None) -> list[Finding]:
    out: list[Finding] = []
    if report is None:
        return out

    if report.insufficient:
        out.append(Finding(
            severity="info",
            category="temporal",
            title="Temporal analysis skipped",
            detail=report.insufficient,
            columns=[report.time_column],
            score=0.0,
        ))

    for sig in report.signals:
        if sig.severity == "none":
            continue
        score = sig.leak_gap or sig.drift_ks or sig.time_monotonicity or 0.0
        # "Future-information" leakage: random PPS holds but the time split
        # collapses. Reason string from `_classify()` carries that signature.
        is_post_event = sig.severity == "critical" and "future information" in (sig.reason or "")
        category = "leakage" if is_post_event else "temporal"
        out.append(Finding(
            severity=sig.severity,
            category=category,
            title=_temporal_title(sig),
            detail=sig.reason,
            columns=[sig.feature, report.time_column] + ([target] if target else []),
            score=float(score),
        ))

    # Target rate drift — informational, one finding
    if target and is_target_drifted(report):
        kind = report.target_drift_kind
        if kind == "binary":
            detail = (
                f"Target rate varies by {report.target_drift:.1%} across time deciles. "
                "Models trained on one period may not generalize."
            )
        elif kind == "multiclass":
            detail = (
                f"Per-class rate varies by up to {report.target_drift:.1%} across time deciles. "
                "Class mix shifts over time; models may not generalize."
            )
        elif kind == "regression_ratio":
            detail = (
                f"Target mean varies by {report.target_drift:.1f}× across time deciles. "
                "Models trained on one period may not generalize."
            )
        else:  # regression_diff (scale-dependent)
            scale_note = ""
            if report.target_drift_score is not None:
                scale_note = f" ({report.target_drift_score:.1f}× target spread)"
            detail = (
                f"Target mean range across deciles is {report.target_drift:.3g} "
                f"{scale_note}. Models trained on one period may not generalize."
            )
        out.append(Finding(
            severity="info",
            category="temporal",
            title=f"`{target}` distribution drifts over `{report.time_column}`",
            detail=detail,
            columns=[target, report.time_column],
            score=min(report.target_drift_score or report.target_drift or 0.0, 1.0),
        ))

    return out


def _temporal_title(sig) -> str:  # TemporalSignal
    if sig.severity == "critical":
        return f"`{sig.feature}` may leak future information"
    if sig.time_monotonicity is not None and sig.time_monotonicity >= 0.95:
        return f"`{sig.feature}` is monotonic with time"
    if sig.drift_ks is not None and sig.drift_ks >= 0.3:
        return f"`{sig.feature}` distribution drifts over time"
    return f"`{sig.feature}` has temporal anomaly"


def rank(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (f.rank, -f.score))
