"""Target column typing shared across profiling, correlations, and temporal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from biopsy.stats import ColumnStats

TargetTaskKind = Literal["classification", "regression"]


def target_task_kind(stats: ColumnStats) -> TargetTaskKind:
    """Whether modeling should treat the target as classification or regression."""
    if (
        stats.kind in {"text", "bool"}
        or (stats.kind == "numeric" and stats.n_unique <= 20)
    ):
        return "classification"
    return "regression"


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
