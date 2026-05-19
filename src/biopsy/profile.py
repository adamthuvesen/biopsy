"""High-level profiling pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from biopsy.clustering import Cluster, ClusterReport, ShortlistEntry, cluster_features
from biopsy.correlations import (
    CorrelationPair,
    TargetSignal,
    correlation_pairs,
    target_signal,
)
from biopsy.findings import (
    Finding,
    column_findings,
    correlation_findings,
    rank,
    target_findings,
    target_summary_findings,
    temporal_findings,
)
from biopsy.io import Source, load
from biopsy.matrix import SampleCache
from biopsy.serialize import to_jsonable
from biopsy.stats import ColumnStats, _quote, compute_all, compute_column
from biopsy.temporal import (
    TemporalReport,
    TemporalSignal,
    TimeBucket,
    _target_kind,
    resolve_time_column,
    temporal_signals,
)

ProgressCallback = Callable[[str], None]


@dataclass
class TargetSummary:
    name: str
    kind: str
    n: int
    n_null: int
    n_unique: int
    class_counts: list[tuple[str, int]] = field(default_factory=list)
    positive_value: str | None = None
    positive_count: int | None = None
    positive_rate: float | None = None
    min_class_count: int | None = None


@dataclass
class Profile:
    """A profiled dataset: column stats, correlations, target signal,
    temporal report, redundancy clusters, and ranked findings.

    Built by `biopsy.profile(...)`. Use `top_findings()`, `leakage_suspects()`,
    `feature_shortlist()`, `drop_candidates()` for the curated views; access
    `columns`, `correlations`, etc. directly for the full payload. Pandas
    callers can use the `*_frame()` helpers.
    """

    source_name: str
    source_path: Path | None
    n_rows: int
    n_cols: int
    elapsed_seconds: float
    target: str | None
    time_column: str | None
    target_summary: TargetSummary | None
    columns: dict[str, ColumnStats]
    correlations: list[CorrelationPair]
    target_signals: list[TargetSignal] = field(default_factory=list)
    temporal: TemporalReport | None = None
    clusters: ClusterReport | None = None
    findings: list[Finding] = field(default_factory=list)

    def findings_records(self) -> list[dict[str, Any]]:
        return [to_jsonable(f) for f in self.findings]

    def columns_records(self) -> list[dict[str, Any]]:
        return [to_jsonable(s) for s in self.columns.values()]

    def target_signal_records(self) -> list[dict[str, Any]]:
        return [to_jsonable(s) for s in self.target_signals]

    def shortlist_records(self) -> list[dict[str, Any]]:
        if self.clusters is None:
            return []
        return [to_jsonable(s) for s in self.clusters.shortlist]

    def action_plan(self) -> Any:
        """Synthesized modeling action plan (drop / impute / encode / transform / review)."""
        from biopsy.action_plan import build_action_plan

        return build_action_plan(self)

    def to_sklearn_pipeline_code(self) -> str:
        """Return a runnable Python module that builds a ColumnTransformer
        wired to this profile's action plan."""
        from biopsy.action_plan import to_sklearn_pipeline_code

        return to_sklearn_pipeline_code(self, self.action_plan())

    def diff(self, other: Profile) -> ProfileDiff:
        """Finding-level diff: appeared / resolved / severity changes,
        schema changes, top-K target-signal rank changes."""
        return diff_profiles(self, other)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | Path) -> Path:
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        return cls(
            source_name=str(data["source_name"]),
            source_path=(
                Path(data["source_path"]) if data.get("source_path") is not None else None
            ),
            n_rows=int(data["n_rows"]),
            n_cols=int(data["n_cols"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
            target=data.get("target"),
            time_column=data.get("time_column"),
            target_summary=_from_target_summary(data.get("target_summary")),
            columns={
                name: _column_stats_from_payload(payload)
                for name, payload in data.get("columns", {}).items()
            },
            correlations=[
                CorrelationPair(**payload) for payload in data.get("correlations", [])
            ],
            target_signals=[
                TargetSignal(**payload) for payload in data.get("target_signals", [])
            ],
            temporal=_from_temporal_report(data.get("temporal")),
            clusters=_from_cluster_report(data.get("clusters")),
            findings=[Finding(**payload) for payload in data.get("findings", [])],
        )

    def _repr_html_(self) -> str:
        from biopsy.render.html import render_string

        return render_string(self, embed_plotly=False)

    def show(self) -> Any:
        """Display the HTML report in a notebook and return this profile."""
        try:
            from IPython.display import HTML, display
        except ImportError:
            return self._repr_html_()
        display(HTML(self._repr_html_()))
        return self

    def top_findings(
        self,
        limit: int | None = 12,
        severity: str | Iterable[str] | None = None,
        category: str | Iterable[str] | None = None,
    ) -> list[Finding]:
        severities = _filter_set(severity)
        categories = _filter_set(category)
        findings = [
            f for f in self.findings
            if (severities is None or f.severity in severities)
            and (categories is None or f.category in categories)
        ]
        return findings if limit is None else findings[:limit]

    def feature_shortlist(
        self,
        limit: int | None = None,
        include_weak: bool = False,
    ) -> list[str]:
        if self.clusters is None:
            return []
        features = [
            e.feature for e in self.clusters.shortlist
            if include_weak or not e.is_weak
        ]
        return features if limit is None else features[:limit]

    def leakage_suspects(self) -> list[str]:
        signal_leaks = [s.feature for s in self.target_signals if s.is_leak_suspect]
        skip = {self.target, self.time_column}
        finding_leaks = [
            c
            for f in self.findings
            if f.category == "leakage" and f.severity == "critical"
            for c in f.columns
            if c not in skip
        ]
        return _unique(signal_leaks + finding_leaks)

    def drop_candidates(self, include_leakage: bool = True) -> list[str]:
        candidates: list[str] = []
        for finding in self.findings:
            if finding.category in {"quality", "suspicious"} and finding.severity in {
                "critical",
                "warning",
            }:
                candidates.extend(c for c in finding.columns if c != self.target)
        if include_leakage:
            candidates.extend(self.leakage_suspects())
        return _unique(candidates)

    def findings_frame(self) -> Any:
        pd = _pandas()
        return pd.DataFrame(self.findings_records())

    def columns_frame(self) -> Any:
        pd = _pandas()
        return pd.DataFrame(self.columns_records())

    def target_signal_frame(self) -> Any:
        pd = _pandas()
        return pd.DataFrame(self.target_signal_records())

    def shortlist_frame(self) -> Any:
        pd = _pandas()
        return pd.DataFrame(self.shortlist_records())


@dataclass
class FindingDiffEntry:
    title: str
    category: str
    severity: str
    columns: list[str]
    detail: str

    @classmethod
    def from_finding(cls, f: Finding) -> FindingDiffEntry:
        return cls(
            title=f.title, category=f.category, severity=f.severity,
            columns=list(f.columns), detail=f.detail,
        )


@dataclass
class SeverityChange:
    title: str
    category: str
    from_severity: str
    to_severity: str
    columns: list[str]


@dataclass
class RankChange:
    feature: str
    from_rank: int | None
    to_rank: int | None
    from_score: float | None
    to_score: float | None


@dataclass
class ProfileDiff:
    """Difference between two profiles at the finding level."""

    a_name: str
    b_name: str
    appeared: list[FindingDiffEntry] = field(default_factory=list)
    resolved: list[FindingDiffEntry] = field(default_factory=list)
    severity_changed: list[SeverityChange] = field(default_factory=list)
    schema_added: list[str] = field(default_factory=list)
    schema_removed: list[str] = field(default_factory=list)
    rank_changed: list[RankChange] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.appeared or self.resolved or self.severity_changed
            or self.schema_added or self.schema_removed or self.rank_changed
        )


def diff_profiles(a: Profile, b: Profile) -> ProfileDiff:
    a_by_key: dict[tuple[str, str], Finding] = {
        _finding_key(f): f for f in a.findings
    }
    b_by_key: dict[tuple[str, str], Finding] = {
        _finding_key(f): f for f in b.findings
    }
    appeared = [FindingDiffEntry.from_finding(f) for k, f in b_by_key.items() if k not in a_by_key]
    resolved = [FindingDiffEntry.from_finding(f) for k, f in a_by_key.items() if k not in b_by_key]
    severity_changed: list[SeverityChange] = []
    for key, fa in a_by_key.items():
        fb = b_by_key.get(key)
        if fb is None:
            continue
        if fa.severity != fb.severity:
            severity_changed.append(SeverityChange(
                title=fb.title,
                category=fb.category,
                from_severity=fa.severity,
                to_severity=fb.severity,
                columns=list(fb.columns),
            ))

    a_cols = set(a.columns)
    b_cols = set(b.columns)
    schema_added = sorted(b_cols - a_cols)
    schema_removed = sorted(a_cols - b_cols)

    a_rank = {s.feature: (i + 1, s.score) for i, s in enumerate(a.target_signals[:30])}
    b_rank = {s.feature: (i + 1, s.score) for i, s in enumerate(b.target_signals[:30])}
    rank_changed: list[RankChange] = []
    for feat in set(a_rank) | set(b_rank):
        ar = a_rank.get(feat)
        br = b_rank.get(feat)
        if ar is None or br is None:
            rank_changed.append(RankChange(
                feature=feat,
                from_rank=ar[0] if ar else None,
                to_rank=br[0] if br else None,
                from_score=ar[1] if ar else None,
                to_score=br[1] if br else None,
            ))
            continue
        if abs(ar[0] - br[0]) >= 3:
            rank_changed.append(RankChange(
                feature=feat,
                from_rank=ar[0], to_rank=br[0],
                from_score=ar[1], to_score=br[1],
            ))
    rank_changed.sort(
        key=lambda c: abs((c.from_rank or 0) - (c.to_rank or 0)),
        reverse=True,
    )

    return ProfileDiff(
        a_name=a.source_name,
        b_name=b.source_name,
        appeared=appeared,
        resolved=resolved,
        severity_changed=severity_changed,
        schema_added=schema_added,
        schema_removed=schema_removed,
        rank_changed=rank_changed[:15],
    )


def _finding_key(f: Finding) -> tuple[str, str]:
    """Stable key for matching findings across profiles.

    Keys on `kind` when present (newly-emitted findings) so a percent change
    in the title (e.g. "30% nulls" → "50% nulls") doesn't split one trend into
    appeared/resolved. Falls back to the title-prefix scheme for older
    round-tripped JSON profiles.
    """
    col = f.columns[0] if f.columns else ""
    if f.kind:
        return (f.category, f"{col}|kind:{f.kind}")
    base = f.title.split("(")[0].rstrip()
    return (f.category, f"{col}|{base}")


def load_profile(path: str | Path) -> Profile:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return Profile.from_dict(data)


def _from_target_summary(data: dict[str, Any] | None) -> TargetSummary | None:
    if data is None:
        return None
    payload = dict(data)
    payload["class_counts"] = [tuple(x) for x in payload.get("class_counts", [])]
    return TargetSummary(**payload)


def _column_stats_from_payload(payload: dict[str, Any]) -> ColumnStats:
    """Coerce JSON-round-tripped lists back into the tuple shapes declared on ColumnStats."""
    p = dict(payload)
    p["top_values"] = [tuple(x) for x in p.get("top_values", [])]
    p["histogram"] = [tuple(x) for x in p.get("histogram", [])]
    p["temporal_buckets"] = [tuple(x) for x in p.get("temporal_buckets", [])]
    return ColumnStats(**p)


def _from_temporal_report(data: dict[str, Any] | None) -> TemporalReport | None:
    if data is None:
        return None
    return TemporalReport(
        time_column=data["time_column"],
        target=data.get("target"),
        signals=[TemporalSignal(**payload) for payload in data.get("signals", [])],
        target_drift=data.get("target_drift"),
        target_drift_kind=data.get("target_drift_kind"),
        insufficient=data.get("insufficient"),
        target_drift_score=data.get("target_drift_score"),
        time_buckets=[
            TimeBucket(**payload) for payload in data.get("time_buckets", [])
        ],
    )


def _from_cluster_report(data: dict[str, Any] | None) -> ClusterReport | None:
    if data is None:
        return None
    return ClusterReport(
        clusters=[Cluster(**payload) for payload in data.get("clusters", [])],
        shortlist=[
            ShortlistEntry(**payload) for payload in data.get("shortlist", [])
        ],
        cutoff=float(data["cutoff"]),
        n_features=int(data.get("n_features", 0)),
        n_singletons=int(data.get("n_singletons", 0)),
    )


def profile(
    data: str | Path | Any,
    target: str | None = None,
    time_col: str | None = None,
    sample: int | None = None,
    hist_bins: int = 24,
    cluster_cutoff: float = 0.30,
    shortlist_size: int | None = None,
    exclude: list[str] | None = None,
    ignore_missing_exclude: bool = False,
    where: list[str] | None = None,
    source_name: str | None = None,
    deep_correlations: bool = True,
    target_permutation: bool = True,
    target_sample_size: int = 30_000,
    stratify_target: bool = True,
    bootstrap: int = 0,
    pps_seeds: int = 1,
    max_cols: int | None = None,
    progress: ProgressCallback | None = None,
) -> Profile:
    _validate_options(
        sample=sample,
        hist_bins=hist_bins,
        cluster_cutoff=cluster_cutoff,
        shortlist_size=shortlist_size,
    )
    t0 = time.perf_counter()
    _progress(progress, "Loading data")
    src: Source = load(
        data,
        sample=sample,
        exclude=exclude,
        ignore_missing_exclude=ignore_missing_exclude,
        where=where,
        source_name=source_name,
    )

    if target is not None and target not in src.columns:
        raise ValueError(
            f"Target column '{target}' not in dataset. Available: {src.columns}"
        )

    _progress(progress, "Computing column statistics")
    stats = compute_all(src, hist_bins=hist_bins)
    sample_cache = SampleCache(src)
    # If max_cols is set, we want to prioritize columns ranked by target
    # signal. Compute the lightweight univariate ranking first when a target
    # is supplied; otherwise fall back to dtype ordering.
    priority_features: list[str] | None = None
    if max_cols is not None and target is not None and deep_correlations:
        _progress(progress, "Pre-ranking columns for pairwise pass")
        priority_features = _univariate_priority(src, stats, target)

    _progress(progress, "Computing correlations")
    corrs = correlation_pairs(
        src,
        stats,
        include_mutual_info=deep_correlations,
        sample_cache=sample_cache,
        max_cols=max_cols,
        priority_features=priority_features,
    )

    target_sigs: list[TargetSignal] = []
    target_summary = None
    if target:
        target_src, target_stats = _target_source_and_stats(
            data=data,
            src=src,
            stats=stats,
            target=target,
            sample=sample,
            hist_bins=hist_bins,
            exclude=exclude,
            ignore_missing_exclude=ignore_missing_exclude,
            where=where,
            source_name=source_name,
        )
        target_summary = _target_summary(target_src, target_stats[target])
        _progress(progress, "Computing target signal")
        target_sigs = target_signal(
            target_src,
            target_stats,
            target,
            max_rows=target_sample_size,
            include_permutation=target_permutation,
            stratify=stratify_target,
            bootstrap=bootstrap,
            pps_seeds=pps_seeds,
        )

    resolved_time, time_info = resolve_time_column(stats, time_col)
    temporal_report = None
    if resolved_time is not None:
        _progress(progress, "Computing temporal checks")
        temporal_report = temporal_signals(src, stats, resolved_time, target=target)

    _progress(progress, "Clustering redundant features")
    clusters_report = cluster_features(
        src, stats,
        target=target,
        target_signals=target_sigs if target else None,
        cutoff=cluster_cutoff,
        max_shortlist=shortlist_size,
        sample_cache=sample_cache,
    )

    _progress(progress, "Ranking findings")
    findings = column_findings(stats, src.n_rows, target=target)
    findings += correlation_findings(corrs)
    if target_summary:
        findings += target_summary_findings(target_summary)
    if target:
        findings += target_findings(target_sigs, target)
    findings += temporal_findings(temporal_report, target)

    if time_info is not None and resolved_time is None:
        findings.append(Finding(
            severity="info",
            category="temporal",
            title="Temporal analysis skipped",
            detail=time_info,
            columns=[],
            score=0.0,
        ))

    findings = rank(findings)

    return Profile(
        source_name=src.source_name,
        source_path=src.source_path,
        n_rows=src.n_rows,
        n_cols=src.n_cols,
        elapsed_seconds=time.perf_counter() - t0,
        target=target,
        time_column=resolved_time,
        target_summary=target_summary,
        columns=stats,
        correlations=corrs,
        target_signals=target_sigs,
        temporal=temporal_report,
        clusters=clusters_report,
        findings=findings,
    )


def _validate_options(
    sample: int | None,
    hist_bins: int,
    cluster_cutoff: float,
    shortlist_size: int | None,
) -> None:
    if sample is not None and sample < 1:
        raise ValueError("sample must be >= 1 when provided.")
    if hist_bins < 1:
        raise ValueError("hist_bins must be >= 1.")
    if not 0 <= cluster_cutoff <= 1:
        raise ValueError("cluster_cutoff must be between 0 and 1.")
    if shortlist_size is not None and shortlist_size < 1:
        raise ValueError("shortlist_size must be >= 1 when provided.")


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _univariate_priority(
    src: Source, stats: dict[str, ColumnStats], target: str,
) -> list[str]:
    """Rank candidate features cheaply for the pairwise-MI cap.

    Numeric × numeric pairs use DuckDB's corr() in a single SQL pass; everything
    else falls back to non-null unique count as a coarse "informativeness"
    proxy. Returns column names ordered best → worst.
    """
    target_stats = stats.get(target)
    if target_stats is None:
        return [n for n in stats if n != target]
    eligible = [
        n for n, s in stats.items()
        if n != target and not s.is_constant
    ]
    if not eligible:
        return []
    target_numeric = target_stats.kind == "numeric"
    numeric_eligible = [n for n in eligible if stats[n].kind == "numeric"]
    abs_corr: dict[str, float] = {}
    if target_numeric and numeric_eligible:
        qtarget = _quote(target)
        select = ", ".join(
            f"abs(corr({_quote(n)}, {qtarget}))" for n in numeric_eligible
        )
        row = src.con.execute(f"SELECT {select} FROM data").fetchone()
        for name, value in zip(numeric_eligible, row, strict=True):
            if value is not None:
                abs_corr[name] = float(value)

    def score(name: str) -> tuple[float, int]:
        if name in abs_corr:
            return (abs_corr[name], stats[name].n_unique)
        s = stats[name]
        nonnull = max(s.n - s.n_null, 1)
        # Coarse informativeness fallback for non-numeric and unrankable columns.
        return (0.0, s.n_unique if s.n_unique < nonnull else 0)

    return sorted(eligible, key=score, reverse=True)


def _target_source_and_stats(
    data: str | Path | Any,
    src: Source,
    stats: dict[str, ColumnStats],
    target: str,
    sample: int | None,
    hist_bins: int,
    exclude: list[str] | None,
    ignore_missing_exclude: bool,
    where: list[str] | None,
    source_name: str | None,
) -> tuple[Source, dict[str, ColumnStats]]:
    if sample is None or not isinstance(data, str | Path):
        return src, stats

    target_src = load(
        data,
        exclude=exclude,
        ignore_missing_exclude=ignore_missing_exclude,
        where=where,
        source_name=source_name,
    )
    target_stats = dict(stats)
    if target in target_src.columns:
        target_stats[target] = compute_column(target_src, target, hist_bins=hist_bins)
    return target_src, target_stats


def _target_summary(src: Source, target_stats: ColumnStats) -> TargetSummary:
    kind = _target_kind(target_stats)
    class_counts: list[tuple[str, int]] = []
    positive_value = None
    positive_count = None
    positive_rate = None
    min_class_count = None

    if kind == "classification":
        rows = src.con.execute(f"""
            SELECT {_quote(target_stats.name)}::VARCHAR AS value, COUNT(*) AS c
            FROM data
            WHERE {_quote(target_stats.name)} IS NOT NULL
            GROUP BY 1
            ORDER BY value
        """).fetchall()
        class_counts = [(str(v), int(c)) for v, c in rows]
        if class_counts:
            min_class_count = min(c for _v, c in class_counts)
        if len(class_counts) == 2:
            positive_value, positive_count = class_counts[-1]
            positive_rate = positive_count / max(target_stats.n - target_stats.n_null, 1)

    return TargetSummary(
        name=target_stats.name,
        kind=kind,
        n=target_stats.n,
        n_null=target_stats.n_null,
        n_unique=target_stats.n_unique,
        class_counts=class_counts,
        positive_value=positive_value,
        positive_count=positive_count,
        positive_rate=positive_rate,
        min_class_count=min_class_count,
    )


def _filter_set(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Profile frame helpers require pandas. Install with "
            "`pip install 'biopsy[dataframe]'`."
        ) from exc
    return pd
