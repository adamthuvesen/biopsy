"""Profile JSON load helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biopsy.profile.model import Profile

from biopsy.clustering import Cluster, ClusterReport, ShortlistEntry
from biopsy.stats import ColumnStats
from biopsy.targets import TargetSummary
from biopsy.temporal import TemporalReport, TimeBucket, temporal_signal_from_payload


def load_profile(path: str | Path) -> Profile:
    from biopsy.profile.model import Profile

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
        signals=[temporal_signal_from_payload(payload) for payload in data.get("signals", [])],
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


