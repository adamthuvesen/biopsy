"""High-level profiling pipeline."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from sketch.serialize import to_jsonable
from sketch.stats import ColumnStats, compute_all
from sketch.temporal import TemporalReport, resolve_time_column, temporal_signals


@dataclass
class Profile:
    source_name: str
    source_path: Path | None
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


def profile(
    data: str | Path | Any | None = None,
    target: str | None = None,
    time_col: str | None = None,
    sample: int | None = None,
    hist_bins: int = 24,
    cluster_cutoff: float = 0.30,
    shortlist_size: int | None = None,
    exclude: list[str] | None = None,
    where: list[str] | None = None,
    source_name: str | None = None,
    path: str | Path | Any | None = None,
) -> Profile:
    if data is None:
        if path is None:
            raise TypeError("profile() missing required argument: 'data'")
        data = path
    elif path is not None:
        raise TypeError("Pass either 'data' or 'path', not both.")

    t0 = time.perf_counter()
    src: Source = load(data, sample=sample, exclude=exclude, where=where, source_name=source_name)

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
        source_name=src.source_name,
        source_path=src.source_path,
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
            "`pip install 'sketch-eda[dataframe]'`."
        ) from exc
    return pd
