"""Profiling pipeline orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from biopsy.clustering import cluster_features
from biopsy.correlations import TargetSignal, correlation_pairs, target_signal
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
from biopsy.profile.model import Profile
from biopsy.stats import ColumnStats, _quote, compute_all, compute_column
from biopsy.targets import TargetSummary, target_task_kind
from biopsy.temporal import resolve_time_column, temporal_signals

ProgressCallback = Callable[[str], None]


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
    credentials_env: str | None = None,
    progress: ProgressCallback | None = None,
) -> Profile:
    _validate_options(
        sample=sample,
        hist_bins=hist_bins,
        cluster_cutoff=cluster_cutoff,
        shortlist_size=shortlist_size,
        target_sample_size=target_sample_size,
        bootstrap=bootstrap,
        pps_seeds=pps_seeds,
        max_cols=max_cols,
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
        credentials_env=credentials_env,
    )
    target_src: Source | None = None
    try:
        if target is not None and target not in src.columns:
            raise ValueError(f"Target column '{target}' not in dataset. Available: {src.columns}")

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
        target_src = src
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
            src,
            stats,
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
            findings.append(
                Finding(
                    severity="info",
                    category="temporal",
                    title="Temporal analysis skipped",
                    detail=time_info,
                    columns=[],
                    score=0.0,
                )
            )

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
            source_uri=src.source_uri,
        )
    finally:
        if target_src is not None and target_src.con is not src.con:
            target_src.con.close()
        src.con.close()


def _validate_options(
    sample: int | None,
    hist_bins: int,
    cluster_cutoff: float,
    shortlist_size: int | None,
    target_sample_size: int,
    bootstrap: int,
    pps_seeds: int,
    max_cols: int | None,
) -> None:
    if sample is not None and sample < 1:
        raise ValueError("sample must be >= 1 when provided.")
    if hist_bins < 1:
        raise ValueError("hist_bins must be >= 1.")
    if not 0 <= cluster_cutoff <= 1:
        raise ValueError("cluster_cutoff must be between 0 and 1.")
    if shortlist_size is not None and shortlist_size < 1:
        raise ValueError("shortlist_size must be >= 1 when provided.")
    if target_sample_size < 1:
        raise ValueError("target_sample_size must be >= 1.")
    if bootstrap < 0:
        raise ValueError("bootstrap must be >= 0.")
    if pps_seeds < 1:
        raise ValueError("pps_seeds must be >= 1.")
    if max_cols is not None and max_cols < 2:
        raise ValueError("max_cols must be >= 2 when provided.")


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _univariate_priority(
    src: Source,
    stats: dict[str, ColumnStats],
    target: str,
) -> list[str]:
    """Rank candidate features cheaply for the pairwise-MI cap.

    Numeric × numeric pairs use DuckDB's corr() in a single SQL pass; everything
    else falls back to non-null unique count as a coarse "informativeness"
    proxy. Returns column names ordered best → worst.
    """
    target_stats = stats.get(target)
    if target_stats is None:
        return [n for n in stats if n != target]
    eligible = [n for n, s in stats.items() if n != target and not s.is_constant]
    if not eligible:
        return []
    target_numeric = target_stats.kind == "numeric"
    numeric_eligible = [n for n in eligible if stats[n].kind == "numeric"]
    abs_corr: dict[str, float] = {}
    if target_numeric and numeric_eligible:
        qtarget = _quote(target)
        select = ", ".join(f"abs(corr({_quote(n)}, {qtarget}))" for n in numeric_eligible)
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
    # A warehouse URI is a string but re-loading would re-pull the whole
    # remote table just to compute target metrics. Treat URIs like
    # in-memory frames here: target metrics use the sampled source, which
    # is what the user opted into.
    if isinstance(data, str) and src.source_uri is not None:
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
    kind = target_task_kind(target_stats)
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
