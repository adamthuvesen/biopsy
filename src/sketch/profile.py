"""High-level profiling pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from sketch.clustering import ClusterReport, cluster_features
from sketch.correlations import (
    CorrelationPair,
    TargetSignal,
    correlation_pairs,
    target_signal,
)
from sketch.findings import (
    Finding,
    column_findings,
    correlation_findings,
    rank,
    target_findings,
    temporal_findings,
)
from sketch.io import Source, load
from sketch.stats import ColumnStats, compute_all
from sketch.temporal import TemporalReport, resolve_time_column, temporal_signals


@dataclass
class Profile:
    source_path: Path
    n_rows: int
    n_cols: int
    elapsed_seconds: float
    target: str | None
    time_column: str | None
    columns: dict[str, ColumnStats]
    correlations: list[CorrelationPair]
    target_signals: list[TargetSignal] = field(default_factory=list)
    temporal: TemporalReport | None = None
    clusters: ClusterReport | None = None
    findings: list[Finding] = field(default_factory=list)


def profile(
    path: str | Path,
    target: str | None = None,
    time_col: str | None = None,
    sample: int | None = None,
    hist_bins: int = 24,
    cluster_cutoff: float = 0.30,
    shortlist_size: int | None = None,
    exclude: list[str] | None = None,
    where: list[str] | None = None,
) -> Profile:
    t0 = time.perf_counter()
    src: Source = load(path, sample=sample, exclude=exclude, where=where)

    if target is not None and target not in src.columns:
        raise ValueError(
            f"Target column '{target}' not in dataset. Available: {src.columns}"
        )

    stats = compute_all(src, hist_bins=hist_bins)
    corrs = correlation_pairs(src, stats)

    target_sigs: list[TargetSignal] = []
    if target:
        target_sigs = target_signal(src, stats, target)

    resolved_time, time_info = resolve_time_column(stats, time_col)
    temporal_report = None
    if resolved_time is not None:
        temporal_report = temporal_signals(src, stats, resolved_time, target=target)

    clusters_report = cluster_features(
        src, stats,
        target=target,
        target_signals=target_sigs if target else None,
        cutoff=cluster_cutoff,
        max_shortlist=shortlist_size,
    )

    findings = column_findings(stats, src.n_rows)
    findings += correlation_findings(corrs)
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
        source_path=src.path,
        n_rows=src.n_rows,
        n_cols=src.n_cols,
        elapsed_seconds=time.perf_counter() - t0,
        target=target,
        time_column=resolved_time,
        columns=stats,
        correlations=corrs,
        target_signals=target_sigs,
        temporal=temporal_report,
        clusters=clusters_report,
        findings=findings,
    )
