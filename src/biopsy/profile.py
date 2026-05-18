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
                name: ColumnStats(**payload)
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
        return [s.feature for s in self.target_signals if s.is_leak_suspect]

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


def load_profile(path: str | Path) -> Profile:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return Profile.from_dict(data)


def _from_target_summary(data: dict[str, Any] | None) -> TargetSummary | None:
    if data is None:
        return None
    return TargetSummary(**data)


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
    _progress(progress, "Computing correlations")
    corrs = correlation_pairs(
        src,
        stats,
        include_mutual_info=deep_correlations,
        sample_cache=sample_cache,
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
    kind = (
        "classification" if (
            target_stats.kind in {"text", "bool"} or
            (target_stats.kind == "numeric" and target_stats.n_unique <= 20)
        ) else "regression"
    )
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
