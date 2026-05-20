"""Rich-powered terminal report."""

from __future__ import annotations

from rich import box
from rich.console import Console, Group
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from biopsy.profile import Profile
from biopsy.sparkline import sparkline
from biopsy.stats import ColumnStats

_MAX_WIDTH = 100

SEVERITY_STYLE = {
    "critical": "bold red",
    "warning":  "bold yellow",
    "info":     "dim",
}
SEVERITY_ICON = {"critical": "■", "warning": "▲", "info": "·"}

_BOX = box.SIMPLE_HEAD


def render(
    prof: Profile,
    console: Console | None = None,
    *,
    all_columns: bool = True,
    max_columns: int = 30,
) -> None:
    if console is None:
        console = Console(width=min(Console().width, _MAX_WIDTH))

    console.print(_header(prof))

    if prof.findings:
        console.print(Rule("[bold]Findings[/bold]", style="bright_black"))
        console.print(_findings_block(prof))

    plan = prof.action_plan()
    if _action_plan_has_content(plan):
        target_chip = f" [dim]→[/dim] [yellow]{prof.target}[/yellow]" if prof.target else ""
        console.print(Rule(f"[bold]Action plan[/bold]{target_chip}", style="bright_black"))
        console.print(_action_plan_block(prof, plan, all_columns=all_columns))

    if prof.target_signals:
        console.print(Rule(
            f"[bold]Target signal[/bold] [dim]→[/dim] [yellow]{prof.target}[/yellow]",
            style="bright_black",
        ))
        console.print(_target_table(prof))

    if prof.temporal is not None and prof.temporal.signals:
        console.print(Rule(
            f"[bold]Temporal[/bold] [dim]→[/dim] [yellow]{prof.temporal.time_column}[/yellow]",
            style="bright_black",
        ))
        console.print(_temporal_table(prof))
    elif prof.temporal is not None and prof.temporal.time_buckets:
        console.print(Rule(
            f"[bold]Temporal buckets[/bold] [dim]→[/dim] [yellow]{prof.temporal.time_column}[/yellow]",
            style="bright_black",
        ))
        console.print(_temporal_buckets_table(prof))

    if prof.clusters is not None and prof.clusters.shortlist:
        n_short = len(prof.clusters.shortlist)
        n_clust = len(prof.clusters.clusters)
        cutoff = f"|ρ|≥{1 - prof.clusters.cutoff:.2f}"
        console.print(Rule(
            f"[bold]Feature shortlist[/bold] [dim]· {n_short} of {n_clust} clusters ({cutoff})[/dim]",
            style="bright_black",
        ))
        console.print(_shortlist_table(prof))

    console.print(Rule("[bold]Columns[/bold]", style="bright_black"))
    console.print(_columns_table(prof, all_columns=all_columns, max_columns=max_columns))

    if prof.correlations:
        console.print(Rule("[bold]Top correlations[/bold]", style="bright_black"))
        console.print(_correlations_table(prof))

    console.print(_footer(prof))


# ── Header ────────────────────────────────────────────────────────────────────

def _header(prof: Profile) -> Text:
    out = Text()
    out.append("biopsy", style="bold magenta")
    out.append("  ")
    out.append(prof.source_name, style="")
    out.append("   ")
    out.append(f"{prof.n_rows:,}", style="bold")
    out.append(" rows", style="dim")
    out.append(" × ", style="dim")
    out.append(f"{prof.n_cols}", style="bold")
    out.append(" cols", style="dim")
    if prof.target:
        out.append("   target ", style="dim")
        out.append(prof.target, style="bold yellow")
    out.append(f"   {prof.elapsed_seconds:.1f}s", style="dim")
    out.append("\n")
    return out


# ── Findings ──────────────────────────────────────────────────────────────────

def _findings_block(prof: Profile) -> Text:
    out = Text()
    shown = 0
    for f in prof.findings:
        if shown >= 12 and f.severity == "info":
            break
        icon = SEVERITY_ICON[f.severity]
        style = SEVERITY_STYLE[f.severity]
        out.append(f" {icon}  ", style=style)
        out.append(f"{f.title}\n", style=style)
        if f.detail and f.severity != "info":
            for line in f.detail.splitlines():
                out.append(f"    {line}\n", style="dim")
        shown += 1

    remaining = len(prof.findings) - shown
    if remaining > 0:
        out.append(f"    … and {remaining} more\n", style="dim italic")
    return out


# ── Target signal ─────────────────────────────────────────────────────────────

def _target_table(prof: Profile) -> Table:
    t = Table(box=_BOX, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("feature", style="cyan", no_wrap=True)
    t.add_column("pps",  justify="right")
    t.add_column("mi",   justify="right")
    t.add_column("ρ",    justify="right")
    t.add_column("auc",  justify="right")
    t.add_column("bar",  no_wrap=True)
    t.add_column("note", style="dim", min_width=14)

    for s in prof.target_signals[:15]:
        bar_width = 14
        filled = int(s.score * bar_width)
        bar_color = "red" if s.is_leak_suspect else "green"
        bar = f"[{bar_color}]" + "█" * filled + "[/]" + "·" * (bar_width - filled)
        note = "[red]leakage suspect[/red]" if s.is_leak_suspect else s.method
        t.add_row(
            s.feature,
            f"{s.score:.2f}",
            f"{s.mutual_info:.2f}",
            _fmt_signed(s.spearman),
            _fmt_unsigned(s.raw_auc),
            bar,
            note,
        )
    return t


# ── Temporal ──────────────────────────────────────────────────────────────────

def _temporal_table(prof: Profile) -> Table:
    report = prof.temporal
    assert report is not None
    t = Table(box=_BOX, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("feature",      style="cyan", no_wrap=True)
    t.add_column("random→time",  justify="right")
    t.add_column("drift",        justify="right")
    t.add_column("monotonicity", justify="right")
    t.add_column("note",         overflow="fold")

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
        mono  = f"{s.time_monotonicity:.2f}" if s.time_monotonicity is not None else "—"
        sev_color = {"critical": "red", "warning": "yellow", "info": "cyan"}.get(s.severity, "white")
        t.add_row(s.feature, split, drift, mono, f"[{sev_color}]{s.reason}[/]")
    return t


def _temporal_buckets_table(prof: Profile) -> Table:
    report = prof.temporal
    assert report is not None
    t = Table(box=_BOX, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("time",             style="cyan")
    t.add_column("rows",             justify="right")
    t.add_column("target n",         justify="right")
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
    return t


# ── Feature shortlist ─────────────────────────────────────────────────────────

def _shortlist_table(prof: Profile) -> Table:
    rep = prof.clusters
    assert rep is not None
    t = Table(box=_BOX, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("#",       justify="right", style="dim", width=3)
    t.add_column("feature", style="cyan", no_wrap=True)
    t.add_column("cluster", justify="right", style="dim")
    t.add_column("size",    justify="right", style="dim")
    t.add_column("score",   justify="right")
    t.add_column("method",  style="dim")

    for i, entry in enumerate(rep.shortlist[:30], 1):
        feat = entry.feature
        score_str = f"{entry.score:.2f}"
        if entry.is_weak:
            feat = f"[yellow]{feat}[/yellow]"
            score_str = f"[yellow]{score_str}[/yellow]"
        t.add_row(str(i), feat, f"c{entry.cluster_id}", str(entry.cluster_size), score_str, entry.score_method)
    return t


# ── Columns ───────────────────────────────────────────────────────────────────

def _columns_table(prof: Profile, *, all_columns: bool, max_columns: int) -> Table:
    t = Table(box=_BOX, show_header=True, header_style="bold", expand=True, padding=(0, 1))
    t.add_column("column",            style="cyan", no_wrap=True)
    t.add_column("type",              style="dim")
    t.add_column("null",              justify="right")
    t.add_column("unique",            justify="right")
    t.add_column("distribution / top", overflow="fold", min_width=16)
    t.add_column("summary",           overflow="fold")

    columns = list(prof.columns.values()) if all_columns else _selected_columns(prof, max_columns)
    for s in columns:
        null_str = _pct(s.null_rate)
        if s.null_rate > 0.5:
            null_str = f"[red]{null_str}[/red]"
        elif s.null_rate > 0.1:
            null_str = f"[yellow]{null_str}[/yellow]"

        if s.kind == "numeric":
            spark   = _spark_for(s, width=16)
            summary = _numeric_summary(s)
        elif s.kind in {"text", "bool"}:
            spark   = _categorical_preview(s)
            summary = _cat_summary(s)
        elif s.kind == "temporal":
            spark   = _temporal_preview(s)
            summary = ""
        else:
            spark   = ""
            summary = s.dtype

        t.add_row(s.name, s.dtype.lower(), null_str, f"{s.n_unique:,}", spark, summary)
    return t


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


# ── Correlations ──────────────────────────────────────────────────────────────

def _correlations_table(prof: Profile) -> Table:
    t = Table(box=_BOX, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("a",           style="cyan", no_wrap=True)
    t.add_column("b",           style="cyan", no_wrap=True)
    t.add_column("pearson",     justify="right")
    t.add_column("mutual info", justify="right")
    t.add_column("kind",        style="dim")

    for p in prof.correlations[:12]:
        if p.score < 0.3:
            break
        r    = f"{p.pearson:+.2f}" if p.pearson is not None else "—"
        mi   = f"{p.mutual_info:.2f}" if p.mutual_info is not None else "—"
        kind = "[magenta]non-linear[/magenta]" if p.is_nonlinear else "linear"
        t.add_row(p.a, p.b, r, mi, kind)
    return t


# ── Action plan ───────────────────────────────────────────────────────────────

def _action_plan_has_content(plan) -> bool:
    return bool(
        plan.drop or plan.transform or plan.impute or plan.encode or plan.review
        or plan.split or plan.cv or plan.class_strategy
    )


def _action_plan_block(prof: Profile, plan, *, all_columns: bool) -> Group:
    limit = None if all_columns else 6

    t = Table(box=_BOX, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("type",   style="dim",  no_wrap=True, width=10)
    t.add_column("column", style="cyan", no_wrap=True)
    t.add_column("action")
    t.add_column("reason", overflow="fold")

    _ACTION_STYLE = {
        "drop":      "red",
        "transform": "yellow",
        "impute":    "cyan",
        "encode":    "magenta",
        "review":    "yellow",
    }

    sections = [
        ("drop",      plan.drop),
        ("transform", plan.transform),
        ("impute",    plan.impute),
        ("encode",    plan.encode),
        ("review",    plan.review),
    ]
    total_truncated = 0
    for section_name, items in sections:
        if not items:
            continue
        shown = items if limit is None else items[:limit]
        style = _ACTION_STYLE[section_name]
        for item in shown:
            t.add_row(section_name, item.column, f"[{style}]{item.action}[/]", item.reason)
        if limit is not None and len(items) > limit:
            total_truncated += len(items) - limit

    extras = Text()
    if plan.split is not None:
        extras.append("  split  ", style="dim")
        extras.append(plan.split.kind, style="bold")
        extras.append(f" — {plan.split.detail}\n", style="dim")
    if plan.cv is not None:
        extras.append("  cv     ", style="dim")
        extras.append(plan.cv.kind, style="bold")
        extras.append(f" — {plan.cv.detail}\n", style="dim")
    if plan.class_strategy is not None:
        extras.append("  class  ", style="dim")
        extras.append(plan.class_strategy.kind, style="bold")
        extras.append(f" — {plan.class_strategy.detail}\n", style="dim")

    parts: list = [t]
    if total_truncated:
        parts.append(Text(f"  … +{total_truncated} more items", style="dim italic"))
    if extras.plain:
        parts.append(extras)
    return Group(*parts)


# ── Footer ────────────────────────────────────────────────────────────────────

def _footer(prof: Profile) -> Text:
    return Text(
        f"\nProfiled {prof.n_rows:,} rows × {prof.n_cols} cols in {prof.elapsed_seconds:.2f}s",
        style="dim",
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _spark_for(s: ColumnStats, width: int = 16) -> str:
    counts = [c for _lo, _hi, c in s.histogram]
    return sparkline(counts, width=width) if counts else ""


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
        if len(label) > 14:
            label = label[:13] + "…"
        items.append(f"{label} [dim]({c:,})[/dim]")
    return "  ".join(items)


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


def _fmt_signed(x: float | None) -> str:
    return "—" if x is None else f"{x:+.2f}"


def _fmt_unsigned(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


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
