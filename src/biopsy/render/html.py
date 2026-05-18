"""HTML report — single self-contained file with Plotly charts.

Design language: forensic editorial. Cream paper, warm ink, oxblood accent.
Distinctive italic serif display type; tabular monospace numerals; ruled
hairlines. The report should read like a clinical chart, not a dashboard.

Charts share a single Plotly template; no gridline clutter, no chart junk.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
from plotly.offline import get_plotlyjs

from biopsy.correlations import CorrelationPair, TargetSignal
from biopsy.findings import Finding
from biopsy.profile import Profile
from biopsy.stats import ColumnStats
from biopsy.temporal import TemporalReport, TemporalSignal

# --- palette (forensic editorial) ------------------------------------------
# Warm paper, deep stone ink, oxblood accent. Severity colors stay vivid.

INK = "#1C1917"        # stone-900 — primary text
INK_2 = "#44403C"      # stone-700
INK_3 = "#78716C"      # stone-500
INK_4 = "#A8A29E"      # stone-400 (faint labels)
SURFACE = "#F7F3EB"    # warm cream paper
SURFACE_2 = "#EFEAE0"  # paper shadow
LINE = "#E5DDD0"       # hairline rule
LINE_2 = "#D6CDBC"     # stronger rule
ACCENT = "#7C2D12"     # oxblood — primary accent
ACCENT_SOFT = "#FBEEE6"
ACCENT_DEEP = "#431407"
WARN = "#A16207"       # muted ochre
CRIT = "#9F1239"       # rose-800
OK = "#166534"         # forest

# Dark mode tokens (resolved client-side via CSS vars, but used here for charts).
DARK_INK = "#F5F0E6"
DARK_SURFACE = "#1A1714"
DARK_LINE = "#3A332C"

SEVERITY_COLOR = {"critical": CRIT, "warning": WARN, "info": INK_2}

# Editorial colorway: oxblood, indigo, sage, ochre, plum, teal.
COLORWAY = ["#7C2D12", "#1E3A8A", "#3F6212", "#A16207", "#6B21A8", "#0E7490"]


# --- plotly template -------------------------------------------------------

_TEMPLATE_NAME = "biopsy"


def _ensure_template() -> None:
    # Plotly stores templates in a process-global registry. Register once,
    # lazily, only when an HTML report is actually rendered — importing this
    # module shouldn't mutate other libraries' Plotly defaults.
    if _TEMPLATE_NAME in pio.templates:
        return
    pio.templates[_TEMPLATE_NAME] = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family=(
                    "'Geist', ui-sans-serif, -apple-system, BlinkMacSystemFont, "
                    "'Segoe UI', sans-serif"
                ),
                size=12,
                color=INK_2,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(
                showgrid=False, showline=True, linecolor=LINE_2,
                ticks="outside", tickcolor=LINE_2,
                tickfont=dict(color=INK_3, size=11),
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=True, gridcolor=LINE, gridwidth=1,
                showline=False, zeroline=False,
                tickfont=dict(color=INK_3, size=11),
            ),
            colorway=COLORWAY,
            hoverlabel=dict(
                bgcolor=SURFACE, bordercolor=LINE_2,
                font=dict(color=INK, family="'JetBrains Mono', ui-monospace, monospace"),
            ),
        )
    )


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
        opacity=0.92,
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
        template="biopsy", height=180,
        margin=dict(l=30, r=10, t=18, b=28),
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
        opacity=0.92,
        hovertemplate="<b>%{x}</b><br>rows: %{y:,}<extra></extra>",
    )
    fig.update_layout(
        template="biopsy", height=180,
        margin=dict(l=30, r=10, t=18, b=38),
        showlegend=False,
        xaxis_title=None, yaxis_title=None,
    )
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
        opacity=0.92,
        hovertemplate="<b>%{y}</b><br>count: %{x:,}<extra></extra>",
    )
    fig.update_layout(
        template="biopsy",
        height=max(170, 22 * len(labels) + 50),
        margin=dict(l=120, r=20, t=8, b=28),
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
        opacity=0.94,
        hovertemplate="<b>%{y}</b><br>score: %{x:.3f}<extra></extra>",
    )
    fig.update_layout(
        template="biopsy",
        height=max(260, 22 * len(labels) + 60),
        margin=dict(l=160, r=20, t=10, b=30),
        showlegend=False,
        xaxis=dict(range=[0, 1], tickformat=".1f"),
    )
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _shortlist_fig(shortlist: list) -> str:
    if not shortlist:
        return ""
    entries = shortlist[:25][::-1]
    labels = [e.feature for e in entries]
    scores = [e.score for e in entries]
    sizes = [e.cluster_size for e in entries]
    weak = [e.is_weak for e in entries]

    # weak → ochre, multi-member cluster → indigo, singleton → oxblood
    colors = [
        WARN if w else (ACCENT if sz == 1 else "#1E3A8A")
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
        opacity=0.94,
        hovertemplate="<b>%{y}</b><br>score: %{x:.3f}<br>%{customdata}<extra></extra>",
        customdata=hover_text,
    )
    fig.update_layout(
        template="biopsy",
        height=max(280, 22 * len(labels) + 60),
        margin=dict(l=190, r=20, t=10, b=30),
        showlegend=False,
        xaxis=dict(range=[0, max(max(scores), 0.05) * 1.05], tickformat=".2f"),
    )
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _temporal_fig(report: TemporalReport) -> str:
    """Diverging bars: random_pps (right) vs. time_pps (left)."""
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
        opacity=0.94,
        hovertemplate="<b>%{y}</b><br>random PPS: %{x:.3f}<extra></extra>",
    )
    fig.add_bar(
        name="time-ordered",
        x=[-v for v in time_vals],
        y=labels,
        orientation="h",
        marker=dict(color=INK_4, line=dict(width=0)),
        opacity=0.85,
        hovertemplate="<b>%{y}</b><br>time PPS: %{customdata:.3f}<extra></extra>",
        customdata=time_vals,
    )

    fig.update_layout(
        template="biopsy",
        height=max(280, 22 * len(labels) + 80),
        margin=dict(l=160, r=20, t=36, b=40),
        barmode="overlay",
        bargap=0.25,
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
        xaxis=dict(
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["1.0", "0.5", "0", "0.5", "1.0"],
            range=[-1.05, 1.05],
            zeroline=True, zerolinecolor=INK_3, zerolinewidth=1,
        ),
    )
    fig.update_yaxes(automargin=True)
    fig.add_annotation(
        x=-0.5, y=1.14, xref="x", yref="paper",
        text="← time-ordered", showarrow=False,
        font=dict(size=11, color=INK_3, family="'Instrument Serif', serif"),
    )
    fig.add_annotation(
        x=0.5, y=1.14, xref="x", yref="paper",
        text="random CV →", showarrow=False,
        font=dict(size=11, color=INK_3, family="'Instrument Serif', serif"),
    )
    return _div(fig)


def _heatmap_fig(
    corrs: list[CorrelationPair],
    stats: dict[str, ColumnStats],
    kind: str,
    max_features: int,
) -> str:
    """kind: 'pearson' or 'mutual_info'."""
    numeric = [n for n, s in stats.items() if s.kind == "numeric" and not s.is_constant]
    if len(numeric) < 2:
        return ""
    if kind == "mutual_info" and not any(p.mutual_info is not None for p in corrs):
        return ""
    if len(numeric) > max_features:
        selected: list[str] = []
        for p in corrs:
            for name in (p.a, p.b):
                if name in numeric and name not in selected:
                    selected.append(name)
                if len(selected) >= max_features:
                    break
            if len(selected) >= max_features:
                break
        if len(selected) < max_features:
            selected.extend(n for n in numeric if n not in selected)
        numeric = selected[:max_features]

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
        # diverging: indigo → cream → oxblood
        colorscale = [
            [0, "#1E3A8A"], [0.25, "#93C5FD"], [0.5, SURFACE],
            [0.75, "#E7B4A3"], [1, ACCENT_DEEP],
        ]
        zmin, zmax = -1, 1
    else:
        colorscale = [[0, SURFACE], [0.5, "#D6BB7E"], [1, ACCENT_DEEP]]
        zmin, zmax = 0, 1

    fig = go.Figure(data=go.Heatmap(
        z=m, x=numeric, y=numeric,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        xgap=2, ygap=2,
        hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>%{z:.3f}<extra></extra>",
        colorbar=dict(thickness=8, len=0.6, outlinewidth=0, tickfont=dict(size=10)),
    ))
    fig.update_layout(
        template="biopsy",
        height=max(360, 28 * n + 80),
        margin=dict(l=120, r=20, t=20, b=120),
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True, autorange="reversed")
    return _div(fig)


def _target_prevalence_fig(class_counts: list[tuple[str, int]]) -> str:
    """Compact horizontal bar of target class distribution."""
    if not class_counts:
        return ""
    # cap at 12 classes; collapse the long tail
    if len(class_counts) > 12:
        head = class_counts[:11]
        tail_total = sum(c for _, c in class_counts[11:])
        shown = head + [(f"… {len(class_counts) - 11} more", tail_total)]
    else:
        shown = list(class_counts)
    labels = [v for v, _ in shown][::-1]
    counts = [c for _, c in shown][::-1]
    total = sum(counts) or 1
    pcts = [c / total for c in counts]

    fig = go.Figure()
    fig.add_bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color=ACCENT, line=dict(width=0)),
        opacity=0.94,
        text=[f"{p:.1%}" for p in pcts],
        textposition="outside",
        textfont=dict(color=INK_2, size=11),
        hovertemplate="<b>%{y}</b><br>n: %{x:,}<extra></extra>",
        cliponaxis=False,
    )
    fig.update_layout(
        template="biopsy",
        height=max(140, 22 * len(labels) + 40),
        margin=dict(l=100, r=60, t=10, b=24),
        showlegend=False,
        xaxis=dict(showticklabels=False, showline=False, ticks=""),
    )
    fig.update_yaxes(automargin=True)
    return _div(fig)


def _div(fig: go.Figure) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


# --- synthesis: action plan + quality score --------------------------------

@dataclass
class ActionItem:
    name: str
    reason: str
    category: str  # original finding category
    severity: str


@dataclass
class ActionPlan:
    drop: list[ActionItem]
    review: list[ActionItem]
    transform: list[ActionItem]


def _build_action_plan(prof: Profile) -> ActionPlan:
    """Synthesize three buckets from findings: drop / review / transform.

    Drop: empty, constant, near-constant, encoded-null columns, ID-shaped.
    Review: leakage suspects, temporal anomalies, high-null, target issues.
    Transform: heavy skew, IQR outliers — modeling preparation hints.
    """
    drop: dict[str, ActionItem] = {}
    review: dict[str, ActionItem] = {}
    transform: dict[str, ActionItem] = {}

    drop_titles = {
        "is 100% null", "is constant", "is near-constant",
        "looks like an identifier",
    }
    review_categories = {"leakage", "temporal", "target"}

    for f in prof.findings:
        col = f.columns[0] if f.columns else None
        if not col or col == prof.target:
            # don't suggest dropping the target column; leakage suspects
            # also reference [feature, target] — keep only the feature.
            if f.category == "leakage" and len(f.columns) >= 1:
                col = f.columns[0]
            else:
                continue

        item = ActionItem(
            name=col,
            reason=f.title.replace("`", ""),
            category=f.category,
            severity=f.severity,
        )

        if any(t in f.title for t in drop_titles) or (
            f.category == "quality" and f.severity == "critical"
        ):
            drop.setdefault(col, item)
        elif f.category in review_categories and f.severity in {"critical", "warning"}:
            review.setdefault(col, item)
        elif f.category == "distribution":
            transform.setdefault(col, item)
        elif (
            f.category == "quality"
            and f.severity in {"warning", "info"}
            and "encoded nulls" in f.title
        ):
            # encoded null sentinels — review (replace before profiling)
            review.setdefault(col, item)

    return ActionPlan(
        drop=list(drop.values()),
        review=list(review.values()),
        transform=list(transform.values()),
    )


def _quality_score(prof: Profile) -> tuple[int, str]:
    """Single 0-100 dataset health score with a one-word verdict.

    Heuristic, not science: penalize critical findings hardest, warnings less,
    info trivially. Floor at 0.
    """
    crit = sum(1 for f in prof.findings if f.severity == "critical")
    warn = sum(1 for f in prof.findings if f.severity == "warning")
    info = sum(1 for f in prof.findings if f.severity == "info")
    penalty = crit * 15 + warn * 5 + info * 1
    score = max(0, 100 - penalty)
    if crit > 0 or score < 40:
        verdict = "critical"
    elif warn > 2 or score < 70:
        verdict = "fair"
    elif warn or info > 4:
        verdict = "good"
    else:
        verdict = "clean"
    return score, verdict


# --- template binding ------------------------------------------------------

def render_string(
    prof: Profile,
    *,
    embed_plotly: bool = True,
    heatmap_limit: int = 60,
) -> str:
    _ensure_template()
    tpl_dir = Path(__file__).parent.parent / "templates"
    env = Environment(
        loader=FileSystemLoader(tpl_dir),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["pct"] = lambda x: "—" if x == 0 else (f"{x:.0%}" if x >= 0.01 else "<1%")
    env.filters["num"] = _num
    env.filters["commafy"] = lambda x: f"{x:,}"
    env.filters["ticks"] = _ticks_filter

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
        columns_payload.append({
            "stats": s,
            "chart": chart,
            "is_target": s.name == prof.target,
            "is_time": s.name == prof.time_column,
        })

    target_chart = ""
    target_prevalence_chart = ""
    if prof.target_signals:
        target_chart = _target_fig(prof.target_signals, prof.target or "")
    if prof.target_summary and prof.target_summary.class_counts:
        target_prevalence_chart = _target_prevalence_fig(
            prof.target_summary.class_counts
        )

    temporal_chart = ""
    temporal_signals = []
    temporal_buckets = []
    if prof.temporal is not None:
        temporal_buckets = prof.temporal.time_buckets
        if prof.temporal.signals:
            temporal_chart = _temporal_fig(prof.temporal)
            temporal_signals = [s for s in prof.temporal.signals if s.severity != "none"]

    shortlist_chart = ""
    clusters_payload = []
    shortlist_entries = []
    if prof.clusters is not None and prof.clusters.shortlist:
        shortlist_entries = prof.clusters.shortlist
        shortlist_chart = _shortlist_fig(prof.clusters.shortlist)
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

    pearson_heatmap = _heatmap_fig(
        prof.correlations, prof.columns, "pearson", max_features=heatmap_limit,
    )
    mi_heatmap = _heatmap_fig(
        prof.correlations, prof.columns, "mutual_info", max_features=heatmap_limit,
    )

    action_plan = _build_action_plan(prof)
    quality_score, verdict = _quality_score(prof)

    # Severity tallies for the vital signs ribbon.
    sev_counts = {
        "critical": sum(1 for f in prof.findings if f.severity == "critical"),
        "warning": sum(1 for f in prof.findings if f.severity == "warning"),
        "info": sum(1 for f in prof.findings if f.severity == "info"),
    }
    # Quality stats for the vitals: numeric/text/temporal split.
    kind_counts: dict[str, int] = {}
    null_total = 0
    for s in prof.columns.values():
        kind_counts[s.kind] = kind_counts.get(s.kind, 0) + 1
        null_total += s.n_null
    cells_total = prof.n_rows * max(prof.n_cols, 1)
    null_share = (null_total / cells_total) if cells_total else 0.0

    # Verdict cards: top 3 findings (critical, warning, info — best each).
    verdict_cards = _verdict_cards(prof.findings)

    tpl = env.get_template("report.html.j2")
    html = tpl.render(
        prof=prof,
        columns=columns_payload,
        target_chart=target_chart,
        target_prevalence_chart=target_prevalence_chart,
        target_signals=prof.target_signals,
        temporal_chart=temporal_chart,
        temporal_signals=temporal_signals,
        temporal_buckets=temporal_buckets,
        shortlist_chart=shortlist_chart,
        shortlist=shortlist_entries,
        clusters=clusters_payload,
        cluster_cutoff=prof.clusters.cutoff if prof.clusters else None,
        pearson_heatmap=pearson_heatmap,
        mi_heatmap=mi_heatmap,
        action_plan=action_plan,
        quality_score=quality_score,
        verdict=verdict,
        sev_counts=sev_counts,
        kind_counts=kind_counts,
        null_share=null_share,
        verdict_cards=verdict_cards,
        severity_color=SEVERITY_COLOR,
        plotly_cdn=None if embed_plotly else "https://cdn.plot.ly/plotly-2.35.2.min.js",
        plotly_js=get_plotlyjs() if embed_plotly else None,
        palette={
            "ink": INK, "ink_2": INK_2, "ink_3": INK_3, "ink_4": INK_4,
            "surface": SURFACE, "surface_2": SURFACE_2,
            "line": LINE, "line_2": LINE_2,
            "accent": ACCENT, "accent_soft": ACCENT_SOFT, "accent_deep": ACCENT_DEEP,
            "warn": WARN, "crit": CRIT, "ok": OK,
            "dark_ink": DARK_INK, "dark_surface": DARK_SURFACE, "dark_line": DARK_LINE,
        },
    )
    return html


def render(
    prof: Profile,
    output_path: str | Path,
    *,
    embed_plotly: bool = True,
    heatmap_limit: int = 60,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    html = render_string(prof, embed_plotly=embed_plotly, heatmap_limit=heatmap_limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _verdict_cards(findings: list[Finding]) -> list[Finding]:
    """Pick the top finding per severity tier for the verdict callouts.

    At most three cards, in severity order. Skips empty tiers.
    """
    seen: dict[str, Finding] = {}
    for f in findings:
        if f.severity in {"critical", "warning", "info"} and f.severity not in seen:
            seen[f.severity] = f
        if len(seen) == 3:
            break
    return [seen[k] for k in ("critical", "warning", "info") if k in seen]


_BACKTICK_RE = re.compile(r"`([^`]+)`")


def _ticks_filter(value: str) -> Markup:
    """Wrap `backtick-quoted` runs as <code>…</code>, escaping the rest."""
    if value is None:
        return Markup("")
    parts: list[str] = []
    last = 0
    for m in _BACKTICK_RE.finditer(value):
        parts.append(html_lib.escape(value[last : m.start()]))
        parts.append("<code>" + html_lib.escape(m.group(1)) + "</code>")
        last = m.end()
    parts.append(html_lib.escape(value[last:]))
    return Markup("".join(parts))


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
