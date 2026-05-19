"""Rich-powered terminal report."""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from biopsy.profile import Profile
from biopsy.sparkline import sparkline
from biopsy.stats import ColumnStats

SEVERITY_STYLE = {
    "critical": "bold red",
    "warning":  "bold yellow",
    "info":     "cyan",
}
SEVERITY_ICON = {"critical": "🚨", "warning": "⚠", "info": "•"}

CATEGORY_LABEL = {
    "leakage":      "Leakage",
    "suspicious":   "Suspicious",
    "quality":      "Data quality",
    "distribution": "Distribution",
    "correlation":  "Correlation",
    "target":       "Target signal",
    "temporal":     "Temporal",
}


def render(
    prof: Profile,
    console: Console | None = None,
    *,
    all_columns: bool = True,
    max_columns: int = 30,
) -> None:
    console = console or Console()

    console.print(_header(prof))
    if prof.findings:
        console.print(_findings_panel(prof))
    plan = prof.action_plan()
    if _action_plan_has_content(plan):
        console.print(_action_plan_panel(prof, plan, all_columns=all_columns))
    if prof.target_signals:
        console.print(_target_panel(prof))
    if prof.temporal is not None and prof.temporal.signals:
        console.print(_temporal_panel(prof))
    elif prof.temporal is not None and prof.temporal.time_buckets:
        console.print(_temporal_buckets_panel(prof))
    if prof.clusters is not None and prof.clusters.shortlist:
        console.print(_shortlist_panel(prof))
    console.print(_columns_table(prof, all_columns=all_columns, max_columns=max_columns))
    if prof.correlations:
        console.print(_correlations_panel(prof))
    console.print(_footer(prof))


def _header(prof: Profile) -> Panel:
    head = Text()
    head.append("biopsy", style="bold magenta")
    head.append("  ")
    head.append(prof.source_name, style="dim")
    sub = Text.assemble(
        ("rows ", "dim"), (f"{prof.n_rows:,}", "bold"),
        ("   cols ", "dim"), (f"{prof.n_cols}", "bold"),
        ("   profiled in ", "dim"), (f"{prof.elapsed_seconds:.2f}s", "bold green"),
    )
    if prof.target:
        sub.append("   target ", style="dim")
        sub.append(prof.target, style="bold yellow")
    return Panel(Group(head, sub), border_style="magenta", padding=(0, 2))


def _findings_panel(prof: Profile) -> Panel:
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(width=2)
    table.add_column(ratio=2)
    table.add_column(ratio=3)

    shown = 0
    for f in prof.findings:
        if shown >= 12 and f.severity == "info":
            break
        style = SEVERITY_STYLE[f.severity]
        icon = SEVERITY_ICON[f.severity]
        title = Text(f.title, style=style)
        detail = Text(f.detail or "", style="dim")
        detail.append(f"\nwhy: {f.why}", style="dim italic")
        table.add_row(icon, title, detail)
        shown += 1
    remaining = len(prof.findings) - shown
    if remaining > 0:
        table.add_row("", Text(f"… and {remaining} more", style="dim italic"), "")

    return Panel(
        table,
        title="[bold]Top findings[/bold]",
        border_style="bright_black",
        padding=(1, 2),
    )


def _spark_for(s: ColumnStats, width: int = 24) -> str:
    counts = [c for _lo, _hi, c in s.histogram]
    return sparkline(counts, width=width) if counts else ""


def _columns_table(prof: Profile, *, all_columns: bool, max_columns: int) -> Panel:
    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("column", style="cyan", no_wrap=True)
    t.add_column("type", style="dim")
    t.add_column("null", justify="right")
    t.add_column("unique", justify="right")
    t.add_column("distribution / top", overflow="fold")
    t.add_column("summary", overflow="fold")

    columns = list(prof.columns.values()) if all_columns else _selected_columns(prof, max_columns)
    for s in columns:
        null_str = _pct(s.null_rate)
        if s.null_rate > 0.5:
            null_str = f"[red]{null_str}[/red]"
        elif s.null_rate > 0.1:
            null_str = f"[yellow]{null_str}[/yellow]"

        if s.kind == "numeric":
            spark = _spark_for(s)
            summary = _numeric_summary(s)
        elif s.kind in {"text", "bool"}:
            spark = _categorical_preview(s)
            summary = _cat_summary(s)
        elif s.kind == "temporal":
            spark = _temporal_preview(s)
            summary = ""
        else:
            spark = ""
            summary = s.dtype

        t.add_row(
            s.name,
            s.dtype.lower(),
            null_str,
            f"{s.n_unique:,}",
            spark,
            summary,
        )

    title = "[bold]Columns[/bold]"
    if not all_columns:
        title = f"[bold]Columns · selected {len(columns)} of {prof.n_cols}[/bold]"
    return Panel(t, title=title, border_style="bright_black", padding=(0, 1))


def _selected_columns(prof: Profile, max_columns: int) -> list[ColumnStats]:
    names: list[str] = []
    for f in prof.findings[:20]:
        names.extend(c for c in f.columns if c in prof.columns)
    if prof.target:
        names.append(prof.target)
    for s in prof.target_signals[:10]:
        names.append(s.feature)
    if prof.clusters is not None:
        names.extend(e.feature for e in prof.clusters.shortlist[:10])

    seen: set[str] = set()
    selected: list[ColumnStats] = []
    for name in names:
        if name in seen or name not in prof.columns:
            continue
        seen.add(name)
        selected.append(prof.columns[name])
        if len(selected) >= max_columns:
            return selected

    for name, stats in prof.columns.items():
        if name in seen:
            continue
        selected.append(stats)
        if len(selected) >= max_columns:
            break
    return selected


def _numeric_summary(s: ColumnStats) -> str:
    if s.mean is None:
        return ""
    bits = [f"μ={_num(s.mean)}", f"σ={_num(s.std)}", f"[{_num(s.min)}…{_num(s.max)}]"]
    if s.skew is not None and abs(s.skew) > 1:
        color = "red" if abs(s.skew) > 3 else "yellow"
        bits.append(f"[{color}]skew={s.skew:+.2f}[/{color}]")
    if s.n_outliers_iqr:
        bits.append(f"[yellow]{s.n_outliers_iqr:,} outliers[/yellow]")
    return "  ".join(bits)


def _categorical_preview(s: ColumnStats, max_items: int = 3) -> str:
    if not s.top_values:
        return ""
    items = []
    for v, c in s.top_values[:max_items]:
        label = str(v)
        if len(label) > 18:
            label = label[:17] + "…"
        items.append(f"{label} [dim]({c:,})[/dim]")
    return "   ".join(items)


def _temporal_preview(s: ColumnStats) -> str:
    if not s.top_values:
        return ""
    parts = dict(s.top_values)
    return f"{parts.get('min', '?')} → {parts.get('max', '?')}"


def _cat_summary(s: ColumnStats) -> str:
    if s.avg_len is not None:
        return f"avg_len={s.avg_len:.0f}  max_len={s.max_len}"
    if s.is_constant:
        return "[red]constant[/red]"
    return ""


def _target_panel(prof: Profile) -> Panel:
    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("feature", style="cyan")
    t.add_column("pps", justify="right")
    t.add_column("mi", justify="right")
    t.add_column("ρ", justify="right")
    t.add_column("auc", justify="right")
    t.add_column("auc lift", justify="right")
    t.add_column("rel perm", justify="right")
    t.add_column("n", justify="right")
    t.add_column("conf", style="dim")
    t.add_column("bar")
    t.add_column("note", style="dim")

    conf_color = {"low": "red", "medium": "yellow", "high": "green"}
    for s in prof.target_signals[:15]:
        bar_width = 20
        filled = int(s.score * bar_width)
        bar_color = "red" if s.is_leak_suspect else "green"
        bar = f"[{bar_color}]" + "█" * filled + "[/]" + "·" * (bar_width - filled)
        note = "[red]leakage suspect[/red]" if s.is_leak_suspect else s.method
        conf = s.confidence
        conf_disp = f"[{conf_color[conf]}]{conf}[/{conf_color[conf]}]"
        t.add_row(
            s.feature,
            f"{s.score:.2f}",
            f"{s.mutual_info:.2f}",
            _fmt_signed(s.spearman),
            _fmt_unsigned(s.raw_auc),
            _fmt_unsigned(s.auc),
            _fmt_unsigned(s.perm_importance),
            f"{s.support:,}",
            conf_disp,
            bar,
            note,
        )

    return Panel(
        t,
        title=f"[bold]Target signal → [yellow]{prof.target}[/yellow][/bold]",
        border_style="bright_black",
        padding=(0, 1),
    )


def _temporal_panel(prof: Profile) -> Panel:
    report = prof.temporal
    assert report is not None
    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("feature", style="cyan")
    t.add_column("random→time", justify="right")
    t.add_column("drift", justify="right")
    t.add_column("monotonicity", justify="right")
    t.add_column("note", overflow="fold")

    # Sort by severity (critical first), then by leak gap descending
    sev_order = {"critical": 0, "warning": 1, "info": 2, "none": 3}
    sorted_signals = sorted(
        report.signals,
        key=lambda s: (sev_order.get(s.severity, 4), -(s.leak_gap or 0)),
    )

    for s in sorted_signals[:15]:
        if s.severity == "none":
            continue
        if s.random_pps is not None and s.time_pps is not None:
            gap_color = "red" if s.severity == "critical" else (
                "yellow" if s.severity == "warning" else "dim"
            )
            split = f"[{gap_color}]{s.random_pps:.2f} → {s.time_pps:.2f}[/]"
        else:
            split = "—"
        drift = f"{s.drift_ks:.2f}" if s.drift_ks is not None else "—"
        mono = f"{s.time_monotonicity:.2f}" if s.time_monotonicity is not None else "—"
        sev_color = {"critical": "red", "warning": "yellow", "info": "cyan"}.get(
            s.severity, "white"
        )
        note = f"[{sev_color}]{s.reason}[/]"
        t.add_row(s.feature, split, drift, mono, note)

    title = f"[bold]Temporal → [yellow]{report.time_column}[/yellow][/bold]"
    return Panel(t, title=title, border_style="bright_black", padding=(0, 1))


def _temporal_buckets_panel(prof: Profile) -> Panel:
    report = prof.temporal
    assert report is not None
    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("time", style="cyan")
    t.add_column("rows", justify="right")
    t.add_column("target n", justify="right")
    t.add_column("target rate/mean", justify="right")

    for b in report.time_buckets:
        if b.target_rate is not None:
            target_value = f"{b.target_rate:.2%}"
        elif b.target_mean is not None:
            target_value = _num(b.target_mean)
        else:
            target_value = "—"
        t.add_row(
            b.label,
            f"{b.n_rows:,}",
            f"{b.n_target:,}" if b.n_target is not None else "—",
            target_value,
        )

    title = f"[bold]Temporal buckets → [yellow]{report.time_column}[/yellow][/bold]"
    return Panel(t, title=title, border_style="bright_black", padding=(0, 1))


def _shortlist_panel(prof: Profile) -> Panel:
    rep = prof.clusters
    assert rep is not None
    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("#", justify="right", style="dim", width=3)
    t.add_column("feature", style="cyan")
    t.add_column("cluster", justify="right")
    t.add_column("size", justify="right")
    t.add_column("score", justify="right")
    t.add_column("method", style="dim")
    t.add_column("rationale", style="dim", overflow="fold")

    for i, entry in enumerate(rep.shortlist[:30], 1):
        feat = entry.feature
        if entry.is_weak:
            feat = f"[yellow]{feat}[/yellow]"
        score_str = f"{entry.score:.2f}"
        if entry.is_weak:
            score_str = f"[yellow]{score_str}[/yellow]"
        t.add_row(
            str(i),
            feat,
            f"c{entry.cluster_id}",
            str(entry.cluster_size),
            score_str,
            entry.score_method,
            entry.rationale,
        )

    n_clusters = len(rep.clusters)
    title = (
        f"[bold]Feature shortlist[/bold] · "
        f"{len(rep.shortlist)} of {n_clusters} clusters "
        f"(cutoff |ρ|≥{1 - rep.cutoff:.2f})"
    )
    return Panel(t, title=title, border_style="bright_black", padding=(0, 1))


def _fmt_signed(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:+.2f}"


def _fmt_unsigned(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}"


def _correlations_panel(prof: Profile) -> Panel:
    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("a", style="cyan")
    t.add_column("b", style="cyan")
    t.add_column("pearson", justify="right")
    t.add_column("mutual info", justify="right")
    t.add_column("kind", style="dim")

    for p in prof.correlations[:12]:
        if p.score < 0.3:
            break
        r = f"{p.pearson:+.2f}" if p.pearson is not None else "—"
        mi = f"{p.mutual_info:.2f}" if p.mutual_info is not None else "—"
        kind = "[magenta]non-linear[/magenta]" if p.is_nonlinear else "linear"
        t.add_row(p.a, p.b, r, mi, kind)

    return Panel(
        t,
        title="[bold]Top correlations[/bold]",
        border_style="bright_black",
        padding=(0, 1),
    )


def _action_plan_has_content(plan) -> bool:
    return bool(
        plan.drop or plan.transform or plan.impute or plan.encode or plan.review
        or plan.split or plan.cv or plan.class_strategy
    )


def _action_plan_panel(prof: Profile, plan, *, all_columns: bool) -> Panel:
    limit = None if all_columns else 6

    def _table(title: str, items, mark_style: str) -> Table | None:
        if not items:
            return None
        t = Table(
            show_header=True, header_style="bold", border_style="bright_black", expand=True,
        )
        t.add_column(title, style="cyan", no_wrap=True)
        t.add_column("action", style=mark_style)
        t.add_column("reason", overflow="fold")
        shown = items if limit is None else items[:limit]
        for item in shown:
            t.add_row(item.column, item.action, item.reason)
        if limit is not None and len(items) > limit:
            t.add_row("…", "", f"+{len(items) - limit} more")
        return t

    panels: list[Table | Text] = []
    for title, items, style in (
        ("drop", plan.drop, "red"),
        ("transform", plan.transform, "yellow"),
        ("impute", plan.impute, "cyan"),
        ("encode", plan.encode, "magenta"),
        ("review", plan.review, "yellow"),
    ):
        tbl = _table(title, items, style)
        if tbl is not None:
            panels.append(tbl)

    extras = Text()
    if plan.split is not None:
        extras.append("split   ", style="dim")
        extras.append(plan.split.kind, style="bold")
        extras.append(f" — {plan.split.detail}\n", style="dim")
    if plan.cv is not None:
        extras.append("cv      ", style="dim")
        extras.append(plan.cv.kind, style="bold")
        extras.append(f" — {plan.cv.detail}\n", style="dim")
    if plan.class_strategy is not None:
        extras.append("class   ", style="dim")
        extras.append(plan.class_strategy.kind, style="bold")
        extras.append(f" — {plan.class_strategy.detail}\n", style="dim")
    if extras.plain:
        panels.append(extras)

    if not panels:
        panels.append(Text("— nothing to act on.", style="dim italic"))

    target_chip = f" → [yellow]{prof.target}[/yellow]" if prof.target else ""
    return Panel(
        Group(*panels),
        title=f"[bold]Action plan{target_chip}[/bold]",
        border_style="bright_black",
        padding=(0, 1),
    )


def _footer(prof: Profile) -> Text:
    return Text(
        f"Profiled {prof.n_rows:,} rows × {prof.n_cols} cols in {prof.elapsed_seconds:.2f}s",
        style="dim italic",
    )


def _pct(x: float) -> str:
    if x == 0:
        return "—"
    if x < 0.01:
        return "<1%"
    return f"{x:.0%}"


def _num(x: float | None) -> str:
    if x is None:
        return "—"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax < 0.01 or ax >= 1e6:
        return f"{x:.2e}"
    if ax >= 100:
        return f"{x:,.1f}"
    return f"{x:.3g}"
