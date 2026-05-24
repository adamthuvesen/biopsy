"""Finding-level profile diff."""

from __future__ import annotations

from dataclasses import dataclass, field

from biopsy.findings import Finding
from biopsy.profile.model import Profile


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
            title=f.title,
            category=f.category,
            severity=f.severity,
            columns=list(f.columns),
            detail=f.detail,
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
            self.appeared
            or self.resolved
            or self.severity_changed
            or self.schema_added
            or self.schema_removed
            or self.rank_changed
        )


def diff_profiles(a: Profile, b: Profile) -> ProfileDiff:
    a_by_key: dict[tuple[str, str], Finding] = {_finding_key(f): f for f in a.findings}
    b_by_key: dict[tuple[str, str], Finding] = {_finding_key(f): f for f in b.findings}
    appeared = [FindingDiffEntry.from_finding(f) for k, f in b_by_key.items() if k not in a_by_key]
    resolved = [FindingDiffEntry.from_finding(f) for k, f in a_by_key.items() if k not in b_by_key]
    severity_changed: list[SeverityChange] = []
    for key, fa in a_by_key.items():
        fb = b_by_key.get(key)
        if fb is None:
            continue
        if fa.severity != fb.severity:
            severity_changed.append(
                SeverityChange(
                    title=fb.title,
                    category=fb.category,
                    from_severity=fa.severity,
                    to_severity=fb.severity,
                    columns=list(fb.columns),
                )
            )

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
            rank_changed.append(
                RankChange(
                    feature=feat,
                    from_rank=ar[0] if ar else None,
                    to_rank=br[0] if br else None,
                    from_score=ar[1] if ar else None,
                    to_score=br[1] if br else None,
                )
            )
            continue
        if abs(ar[0] - br[0]) >= 3:
            rank_changed.append(
                RankChange(
                    feature=feat,
                    from_rank=ar[0],
                    to_rank=br[0],
                    from_score=ar[1],
                    to_score=br[1],
                )
            )
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
