"""Profile dataclass and query helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from biopsy.clustering import ClusterReport
from biopsy.correlations import CorrelationPair, TargetSignal
from biopsy.findings import Finding
from biopsy.serialize import to_jsonable
from biopsy.stats import ColumnStats
from biopsy.targets import TargetSummary
from biopsy.temporal import TemporalReport

if TYPE_CHECKING:
    from biopsy.action_plan import ActionPlan
    from biopsy.profile.diff import ProfileDiff


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
    # Set for warehouse / object-store sources; mutually exclusive with
    # source_path. Older saved profiles will not have this field — see
    # `from_dict` for the back-compat path.
    source_uri: str | None = None

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

    def action_plan(self) -> ActionPlan:
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
        from biopsy.profile.diff import diff_profiles

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
        from biopsy.profile.serde import (
            _column_stats_from_payload,
            _from_cluster_report,
            _from_target_summary,
            _from_temporal_report,
        )

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
            correlations=[CorrelationPair(**payload) for payload in data.get("correlations", [])],
            target_signals=[TargetSignal(**payload) for payload in data.get("target_signals", [])],
            temporal=_from_temporal_report(data.get("temporal")),
            clusters=_from_cluster_report(data.get("clusters")),
            findings=[Finding(**payload) for payload in data.get("findings", [])],
            source_uri=data.get("source_uri"),
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
            f
            for f in self.findings
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
        features = [e.feature for e in self.clusters.shortlist if include_weak or not e.is_weak]
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


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _filter_set(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Profile frame helpers require pandas. Install with `pip install 'biopsy[dataframe]'`."
        ) from exc
    return pd
