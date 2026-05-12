"""Ranked findings — the opinionated 'what should I look at first' list."""

from __future__ import annotations

from dataclasses import dataclass

from sketch.correlations import CorrelationPair, TargetSignal
from sketch.stats import ColumnStats
from sketch.temporal import TemporalReport, is_target_drifted


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


def column_findings(stats: dict[str, ColumnStats], n_rows: int) -> list[Finding]:
    out: list[Finding] = []

    for s in stats.values():
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
        if s.is_constant:
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
        if s.is_near_constant:
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
        if s.skew is not None and abs(s.skew) > 3:
            out.append(Finding(
                severity="info", category="distribution",
                title=f"`{s.name}` is heavily skewed (skew={s.skew:.1f})",
                detail="Consider log/Box-Cox transform before modeling.",
                columns=[s.name], score=min(abs(s.skew) / 10, 1.0),
            ))

        # outliers
        if s.n_outliers_iqr is not None and s.n_outliers_iqr > 0:
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

        # high-cardinality text (modeling foot-gun)
        if s.kind == "text" and s.n_unique > 0.5 * (s.n - s.n_null) and s.n_unique > 50:
            out.append(Finding(
                severity="info", category="suspicious",
                title=f"`{s.name}` has very high cardinality ({s.n_unique:,} unique)",
                detail="Free-text or near-unique — needs encoding or feature engineering.",
                columns=[s.name], score=0.7,
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
        out.append(Finding(
            severity=sig.severity,
            category="temporal",
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
            detail = (
                f"Target mean range across deciles is {report.target_drift:.3g} "
                f"(target spans negatives or zero). Models may not generalize."
            )
        out.append(Finding(
            severity="info",
            category="temporal",
            title=f"`{target}` distribution drifts over `{report.time_column}`",
            detail=detail,
            columns=[target, report.time_column],
            score=min(report.target_drift or 0.0, 1.0),
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
