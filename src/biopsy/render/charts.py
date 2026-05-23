"""Shared column visualization helpers for HTML (and kind dispatch elsewhere)."""

from __future__ import annotations

from typing import Literal

from biopsy.stats import ColumnStats

ColumnVizKind = Literal["numeric", "categorical", "temporal", "none"]


def column_viz_kind(stats: ColumnStats) -> ColumnVizKind:
    if stats.kind == "numeric":
        return "numeric"
    if stats.kind in {"text", "bool"}:
        return "categorical"
    if stats.kind == "temporal":
        return "temporal"
    return "none"


def column_chart_html(stats: ColumnStats) -> str:
    """Plotly div HTML for one column's distribution chart."""
    # Lazy import avoids circular import with render.html.
    from biopsy.render import html as html_mod

    kind = column_viz_kind(stats)
    if kind == "numeric":
        return html_mod._histogram_fig(stats)
    if kind == "categorical":
        return html_mod._bar_fig(stats)
    if kind == "temporal":
        return html_mod._temporal_column_fig(stats)
    return ""
