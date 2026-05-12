"""HTML report — single self-contained file with Plotly charts.

Design language:
- restrained palette (deep slate ink, off-white surface, indigo accent)
- generous whitespace, monospace numerals, sentence-case titles
- charts share a single template; no gridline clutter, no chart junk
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape

from biopsy.correlations import CorrelationPair, TargetSignal
from biopsy.profile import Profile
from biopsy.stats import ColumnStats
from biopsy.temporal import TemporalReport, TemporalSignal

# --- palette ---------------------------------------------------------------

INK = "#0F172A"        # slate-900 — primary text
INK_2 = "#334155"      # slate-700
INK_3 = "#64748B"      # slate-500
SURFACE = "#FAFAF9"    # off-white
SURFACE_2 = "#F1F5F9"  # slate-100
LINE = "#E2E8F0"       # slate-200
ACCENT = "#4F46E5"     # indigo-600
ACCENT_SOFT = "#EEF2FF"
WARN = "#D97706"       # amber-600
CRIT = "#DC2626"       # red-600
OK = "#059669"         # emerald-600

SEVERITY_COLOR = {"critical": CRIT, "warning": WARN, "info": ACCENT}


# --- plotly template -------------------------------------------------------

def _register_template() -> None:
    pio.templates["biopsy"] = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family="ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Inter', sans-serif",
                size=12,
                color=INK,
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(
                showgrid=False, showline=True, linecolor=LINE,
                ticks="outside", tickcolor=LINE, tickfont=dict(color=INK_3, size=11),
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=True, gridcolor=LINE, gridwidth=1,
                showline=False, zeroline=False,
                tickfont=dict(color=INK_3, size=11),
            ),
            colorway=[ACCENT, "#0891B2", "#7C3AED", "#DB2777", "#65A30D", "#EA580C"],
            hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK)),
        )
    )


_register_template()


# --- chart builders --------------------------------------------------------

def _histogram_fig(s: ColumnStats) -> str:
    if not s.histogram:
        return ""
    centers = [(lo + hi) / 2 for lo, hi, _ in s.histogram]
    widths = [hi - lo for lo, hi, _ in s.histogram]
    counts = [c for *_b, c in s.histogram]

    fig = go.Figure()
    fig.add_bar(
        x=centers, y=counts, width=widths,
        marker=dict(color=ACCENT, line=dict(width=0)),
        hovertemplate="<b>%{x:.4g}</b><br>count: %{y:,}<extra></extra>",
    )
    # quartile markers — draw lines for all three; only label those that are
    # visually separable (at least 8% of x-range apart) to avoid overlap on
    # heavily skewed columns.
    x_range = s.max - s.min if (s.max is not None and s.min is not None and s.max > s.min) else 0
    min_gap = x_range * 0.08
    last_labeled_x: float | None = None
    for q, label in [(s.p25, "Q1"), (s.p50, "median"), (s.p75, "Q3")]:
        if q is None:
            continue
        show_label = last_labeled_x is None or (q - last_labeled_x) >= min_gap
        if show_label:
            fig.add_vline(
                x=q, line_width=1, line_dash="dot", line_color=INK_3,
                annotation_text=label, annotation_position="top",
                annotation_font=dict(size=9, color=INK_3),
            )
            last_labeled_x = q
        else:
            fig.add_vline(x=q, line_width=1, line_dash="dot", line_color=INK_3)
    fig.update_layout(
        template="biopsy", height=200,
        margin=dict(l=30, r=10, t=20, b=30),
        showlegend=False,
        xaxis_title=None, yaxis_title=None,
    )
    return _div(fig)


def _temporal_column_fig(s: ColumnStats) -> str:
    if not s.temporal_buckets:
        return ""
    labels = [b for b, _ in s.temporal_buckets]
    counts = [c for _, c in s.temporal_buckets]

    fig = go.Figure()
    fig.add_bar(
        x=labels, y=counts,
        marker=dict(color=ACCENT, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>rows: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        template="biopsy", height=200,
        margin=dict(l=30, r=10, t=20, b=40),
        showlegend=False,
        xaxis_title=None, yaxis_title=None,
    )
    # Reduce tick density if many buckets
    if len(labels) > 18:
        fig.update_xaxes(nticks=10, tickangle=-30)
    return _div(fig)


def _bar_fig(s: ColumnStats) -> str:
    if not s.top_values:
        return ""
    labels = [str(v)[:24] for v, _ in s.top_values][:12]
    counts = [c for _, c in s.top_values][:12]
    labels = list(reversed(labels))
    counts = list(reversed(counts))

    fig = go.Figure()
    fig.add_bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color=ACCENT, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>count: %{x:,}<extra></extra>",
    )
    fig.update_layout(
        template="biopsy",
        height=max(180, 24 * len(labels) + 60),
        margin=dict(l=120, r=20, t=10, b=30),
        showlegend=False,
        xaxis_title=None, yaxis_title=None,
    )
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _target_fig(signals: list[TargetSignal], target: str) -> str:
    if not signals:
        return ""
    top = signals[:20]
    labels = [s.feature for s in top][::-1]
    scores = [s.score for s in top][::-1]
    colors = [CRIT if s.is_leak_suspect else ACCENT for s in top][::-1]

    fig = go.Figure()
    fig.add_bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>score: %{x:.3f}<extra></extra>",
    )
    fig.update_layout(
        template="biopsy",
        height=max(260, 22 * len(labels) + 60),
        margin=dict(l=140, r=20, t=10, b=30),
        showlegend=False,
        xaxis=dict(range=[0, 1], tickformat=".1f"),
    )
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _shortlist_fig(shortlist: list) -> str:
    """Horizontal bar chart of shortlist scores, color-coded by cluster size."""
    if not shortlist:
        return ""
    entries = shortlist[:25][::-1]
    labels = [e.feature for e in entries]
    scores = [e.score for e in entries]
    sizes = [e.cluster_size for e in entries]
    weak = [e.is_weak for e in entries]

    colors = [
        WARN if w else (ACCENT if sz == 1 else "#7C3AED")  # purple for multi-member clusters
        for w, sz in zip(weak, sizes, strict=True)
    ]
    hover_text = [
        f"cluster c{e.cluster_id} (size {e.cluster_size})"
        for e in entries
    ]

    fig = go.Figure()
    fig.add_bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>score: %{x:.3f}<br>%{customdata}<extra></extra>",
        customdata=hover_text,
    )
    fig.update_layout(
        template="biopsy",
        height=max(280, 22 * len(labels) + 60),
        margin=dict(l=180, r=20, t=10, b=30),
        showlegend=False,
        xaxis=dict(range=[0, max(max(scores), 0.05) * 1.05], tickformat=".2f"),
    )
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _temporal_fig(report: TemporalReport) -> str:
    """Diverging bars: random_pps (right) vs. time_pps (left).

    Critical features (gap exceeds threshold) are highlighted in red.
    """
    sigs: list[TemporalSignal] = [
        s for s in report.signals if s.random_pps is not None and s.time_pps is not None
    ]
    if not sigs:
        return ""
    sigs.sort(key=lambda s: -(s.random_pps - s.time_pps))
    sigs = sigs[:25]
    labels = [s.feature for s in sigs][::-1]
    random_vals = [s.random_pps for s in sigs][::-1]
    time_vals = [s.time_pps for s in sigs][::-1]
    leak = [s.severity == "critical" for s in sigs][::-1]

    fig = go.Figure()
    fig.add_bar(
        name="random CV",
        x=random_vals,
        y=labels,
        orientation="h",
        marker=dict(
            color=[CRIT if lk else ACCENT for lk in leak],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>random PPS: %{x:.3f}<extra></extra>",
    )
    fig.add_bar(
        name="time-ordered",
        x=[-v for v in time_vals],
        y=labels,
        orientation="h",
        marker=dict(color="#94A3B8", line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>time PPS: %{customdata:.3f}<extra></extra>",
        customdata=time_vals,
    )

    fig.update_layout(
        template="biopsy",
        height=max(280, 22 * len(labels) + 80),
        margin=dict(l=150, r=20, t=30, b=40),
        barmode="overlay",
        bargap=0.25,
        showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        xaxis=dict(
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["1.0", "0.5", "0", "0.5", "1.0"],
            range=[-1.05, 1.05],
            zeroline=True, zerolinecolor=INK_3, zerolinewidth=1,
        ),
    )
    fig.update_yaxes(automargin=True)
    fig.add_annotation(
        x=-0.5, y=1.13, xref="x", yref="paper",
        text="← time-ordered", showarrow=False,
        font=dict(size=11, color=INK_3),
    )
    fig.add_annotation(
        x=0.5, y=1.13, xref="x", yref="paper",
        text="random CV →", showarrow=False,
        font=dict(size=11, color=INK_3),
    )
    return _div(fig)


def _heatmap_fig(corrs: list[CorrelationPair], stats: dict[str, ColumnStats], kind: str) -> str:
    """kind: 'pearson' or 'mutual_info'."""
    numeric = [n for n, s in stats.items() if s.kind == "numeric" and not s.is_constant]
    if len(numeric) < 2:
        return ""

    idx = {n: i for i, n in enumerate(numeric)}
    n = len(numeric)
    m = [[None] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 1.0
    for p in corrs:
        if p.a not in idx or p.b not in idx:
            continue
        v = p.pearson if kind == "pearson" else p.mutual_info
        if v is None:
            continue
        m[idx[p.a]][idx[p.b]] = v
        m[idx[p.b]][idx[p.a]] = v

    if kind == "pearson":
        colorscale = [
            [0, "#1E40AF"], [0.25, "#93C5FD"], [0.5, "#F8FAFC"],
            [0.75, "#FCA5A5"], [1, "#991B1B"],
        ]
        zmin, zmax = -1, 1
    else:
        colorscale = [[0, "#F8FAFC"], [0.5, "#A5B4FC"], [1, "#3730A3"]]
        zmin, zmax = 0, 1

    fig = go.Figure(data=go.Heatmap(
        z=m, x=numeric, y=numeric,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>%{z:.3f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.6, outlinewidth=0),
    ))
    fig.update_layout(
        template="biopsy",
        height=max(360, 28 * n + 80),
        margin=dict(l=120, r=20, t=20, b=120),
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True, autorange="reversed")
    return _div(fig)


def _div(fig: go.Figure) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


# --- template binding ------------------------------------------------------

def render(prof: Profile, output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    tpl_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(tpl_dir),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["pct"] = lambda x: "—" if x == 0 else (f"{x:.0%}" if x >= 0.01 else "<1%")
    env.filters["num"] = _num
    env.filters["commafy"] = lambda x: f"{x:,}"

    columns_payload = []
    for s in prof.columns.values():
        if s.kind == "numeric":
            chart = _histogram_fig(s)
        elif s.kind in {"text", "bool"}:
            chart = _bar_fig(s)
        elif s.kind == "temporal":
            chart = _temporal_column_fig(s)
        else:
            chart = ""
        columns_payload.append({"stats": s, "chart": chart})

    target_chart = ""
    if prof.target_signals:
        target_chart = _target_fig(prof.target_signals, prof.target or "")

    temporal_chart = ""
    temporal_signals = []
    if prof.temporal is not None and prof.temporal.signals:
        temporal_chart = _temporal_fig(prof.temporal)
        temporal_signals = [s for s in prof.temporal.signals if s.severity != "none"]

    shortlist_chart = ""
    clusters_payload = []
    shortlist_entries = []
    if prof.clusters is not None and prof.clusters.shortlist:
        shortlist_entries = prof.clusters.shortlist
        shortlist_chart = _shortlist_fig(prof.clusters.shortlist)
        # cluster map: members per cluster, with representative flagged
        rep_set = {e.feature for e in prof.clusters.shortlist}
        clusters_payload = [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "members": c.members,
                "representative": c.representative,
                "mean_abs_correlation": c.mean_abs_correlation,
                "in_shortlist": c.representative in rep_set,
            }
            for c in prof.clusters.clusters
        ]

    pearson_heatmap = _heatmap_fig(prof.correlations, prof.columns, "pearson")
    mi_heatmap = _heatmap_fig(prof.correlations, prof.columns, "mutual_info")

    tpl = env.get_template("report.html.j2")
    html = tpl.render(
        prof=prof,
        columns=columns_payload,
        target_chart=target_chart,
        target_signals=prof.target_signals,
        temporal_chart=temporal_chart,
        temporal_signals=temporal_signals,
        shortlist_chart=shortlist_chart,
        shortlist=shortlist_entries,
        clusters=clusters_payload,
        cluster_cutoff=prof.clusters.cutoff if prof.clusters else None,
        pearson_heatmap=pearson_heatmap,
        mi_heatmap=mi_heatmap,
        severity_color=SEVERITY_COLOR,
        plotly_cdn="https://cdn.plot.ly/plotly-2.35.2.min.js",
        palette={
            "ink": INK, "ink_2": INK_2, "ink_3": INK_3,
            "surface": SURFACE, "surface_2": SURFACE_2, "line": LINE,
            "accent": ACCENT, "accent_soft": ACCENT_SOFT,
            "warn": WARN, "crit": CRIT, "ok": OK,
        },
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _num(x: float | None) -> str:
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 0.01 or ax >= 1e6:
        return f"{x:.2e}"
    if ax >= 100:
        return f"{x:,.2f}"
    return f"{x:.4g}"


# expose JSON helper for tests
def _payload(prof: Profile) -> str:
    return json.dumps({"n_rows": prof.n_rows, "n_cols": prof.n_cols})
