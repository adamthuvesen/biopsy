"""Heuristics for target/time candidates, ID detection, and doctor hints."""

from __future__ import annotations

from biopsy.stats import ColumnStats


def looks_like_id(name: str) -> bool:
    """Heuristic: column name signals an identifier."""
    n = name.lower()
    if n == "id":
        return True
    if n.endswith("_id"):
        return True
    return "uuid" in n


def infer_target(stats: dict[str, ColumnStats]) -> str | None:
    exact = [
        "target",
        "label",
        "y",
        "outcome",
        "churn",
        "churned",
        "converted",
        "conversion",
        "default",
        "fraud",
        "is_fraud",
    ]
    by_lower = {name.lower(): name for name in stats}
    for candidate in exact:
        if candidate in by_lower:
            return by_lower[candidate]
    for name in stats:
        lower = name.lower()
        if lower.endswith("_target") or lower.startswith("target_"):
            return name
    for name, col_stats in stats.items():
        lower = name.lower()
        if lower.startswith(("is_", "has_")) and col_stats.n_unique == 2:
            return name
    return None


def infer_time(stats: dict[str, ColumnStats]) -> str | None:
    temporal = [
        name for name, col_stats in stats.items()
        if col_stats.kind == "temporal" and col_stats.n_unique >= 10
    ]
    if len(temporal) == 1:
        return temporal[0]
    preferred_tokens = ("date", "time", "created", "event", "snapshot", "as_of")
    for name in temporal:
        lower = name.lower()
        if any(token in lower for token in preferred_tokens):
            return name
    return None


def infer_excludes(
    stats: dict[str, ColumnStats],
    *,
    target: str | None,
    time_col: str | None,
) -> list[str]:
    keep = {name for name in (target, time_col) if name}
    excludes: list[str] = []
    for name, col_stats in stats.items():
        if name in keep:
            continue
        non_null = col_stats.n - col_stats.n_null
        if non_null <= 50:
            continue
        if looks_like_id(name) and col_stats.unique_rate >= 0.95:
            excludes.append(name)
            continue
        if (
            col_stats.kind == "text"
            and col_stats.unique_rate >= 0.80
            and col_stats.n_unique > 100
        ):
            excludes.append(name)
    return excludes


def doctor_hints(col_stats: ColumnStats) -> list[str]:
    """Short labels for the doctor table 'looks like' column."""
    hints: list[str] = []
    if looks_like_id(col_stats.name):
        hints.append("identifier")
    if col_stats.kind == "numeric" and col_stats.n_unique <= 2:
        hints.append("boolean")
    if col_stats.kind in {"text", "bool"} and 2 <= col_stats.n_unique <= 20:
        hints.append("low-card categorical")
    if col_stats.kind == "numeric" and 2 < col_stats.n_unique <= 20:
        hints.append("ordinal candidate target")
    if col_stats.kind == "temporal":
        hints.append("time column candidate")
    if col_stats.null_rate >= 0.5:
        hints.append("high-null")
    return hints
