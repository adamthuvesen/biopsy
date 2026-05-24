"""biopsy diff — finding-level diff between saved profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from biopsy.profile import load_profile


def diff_cmd(
    a: Path = typer.Argument(..., help="Side A: saved profile JSON."),
    b: Path = typer.Argument(..., help="Side B: saved profile JSON."),
) -> None:
    """Finding-level diff between two saved profiles."""
    console = Console(width=120)
    prof_a = load_profile(a)
    prof_b = load_profile(b)
    d = prof_a.diff(prof_b)
    print_diff(console, d)


def print_diff(console: Console, d: Any) -> None:
    from rich.panel import Panel
    from rich.table import Table

    console.print(
        Panel(
            f"[bold]Profile diff[/bold] · {d.a_name} → {d.b_name}",
            border_style="magenta",
            padding=(0, 2),
        )
    )
    if d.is_empty():
        console.print("[dim]No differences.[/dim]")
        return

    if d.schema_added or d.schema_removed:
        t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
        t.add_column("change")
        t.add_column("columns", overflow="fold")
        if d.schema_added:
            t.add_row("[green]+ added[/green]", ", ".join(d.schema_added))
        if d.schema_removed:
            t.add_row("[red]- removed[/red]", ", ".join(d.schema_removed))
        console.print(Panel(t, title="[bold]Schema[/bold]", border_style="bright_black"))

    def _findings_table(entries: list[Any], title: str, color: str) -> None:
        if not entries:
            return
        t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
        t.add_column("severity", style="dim")
        t.add_column("category", style="dim")
        t.add_column("title", overflow="fold")
        t.add_column("columns", overflow="fold", style="cyan")
        for e in entries[:15]:
            t.add_row(e.severity, e.category, e.title, ", ".join(e.columns))
        if len(entries) > 15:
            t.add_row("…", "", f"+{len(entries) - 15} more", "")
        console.print(Panel(t, title=f"[{color}]{title}[/{color}]", border_style="bright_black"))

    _findings_table(d.appeared, "Appeared", "yellow")
    _findings_table(d.resolved, "Resolved", "green")

    if d.severity_changed:
        t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
        t.add_column("title", overflow="fold")
        t.add_column("from→to")
        t.add_column("columns", overflow="fold", style="cyan")
        for sc in d.severity_changed[:15]:
            t.add_row(sc.title, f"{sc.from_severity} → {sc.to_severity}", ", ".join(sc.columns))
        console.print(Panel(t, title="[bold]Severity change[/bold]", border_style="bright_black"))

    if d.rank_changed:
        t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
        t.add_column("feature", style="cyan")
        t.add_column("from rank", justify="right")
        t.add_column("to rank", justify="right")
        t.add_column("score Δ", justify="right")
        for rc in d.rank_changed[:15]:
            fr = rc.from_rank if rc.from_rank is not None else "—"
            tr = rc.to_rank if rc.to_rank is not None else "—"
            score_delta = "—"
            if rc.from_score is not None and rc.to_score is not None:
                score_delta = f"{(rc.to_score - rc.from_score):+.2f}"
            t.add_row(rc.feature, str(fr), str(tr), score_delta)
        console.print(
            Panel(t, title="[bold]Target-signal rank change[/bold]", border_style="bright_black")
        )
