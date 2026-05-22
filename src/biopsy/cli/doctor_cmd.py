"""biopsy doctor — quick schema and candidate scan."""

from __future__ import annotations

import typer
from rich.console import Console

import biopsy.cli as cli
from biopsy.cli.common import USER_ERRORS, clean_exit_on_user_error
from biopsy.inference import doctor_hints
from biopsy.stats import ColumnStats
from biopsy.warehouse.doctor import discover_warehouse_schema


def doctor_cmd(
    path: str = typer.Argument(
        ...,
        help=(
            "Data source: file path or warehouse URI "
            "(s3://, https://, postgres://, snowflake://, bigquery://)."
        ),
    ),
    sample: int = typer.Option(5000, "--sample", min=100, help="Rows to scan for inference."),
    credentials_env: str | None = typer.Option(
        None, "--credentials-env",
        help="Prefix for warehouse credential env vars. See `biopsy profile --help`.",
    ),
) -> None:
    """Quick check: schema, candidate targets, candidate time columns.

    Doesn't run the full profile — sub-2-seconds on most datasets.
    Warehouse sources use schema-only discovery and do NOT pull row data.
    """
    console = Console(width=120)
    progress_console = Console(stderr=True, width=120)

    # For warehouse URIs, use cheap schema discovery — no row data
    # transferred. The trade-off: cardinality-based "looks like" hints
    # need row data, so warehouse doctor skips them and prints a tip.
    try:
        stats, display_name, n_rows_estimate, schema_only = doctor_load(
            path, sample=sample, credentials_env=credentials_env,
        )
    except USER_ERRORS as exc:
        clean_exit_on_user_error(exc)
    if schema_only:
        progress_console.print(
            "[dim]biopsy:[/dim] schema-only mode — no row data transferred. "
            "Use `biopsy profile --sample N` for cardinality-based hints."
        )

    from rich.panel import Panel
    from rich.table import Table

    head = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    head.add_column("column", style="cyan", no_wrap=True)
    head.add_column("dtype", style="dim")
    head.add_column("kind", style="dim")
    head.add_column("null", justify="right")
    head.add_column("unique", justify="right")
    head.add_column("looks like", style="yellow")
    for s in stats.values():
        head.add_row(
            s.name, s.dtype.lower(), s.kind,
            f"{s.null_rate:.0%}" if s.null_rate else "—",
            f"{s.n_unique:,}",
            ", ".join(doctor_hints(s)),
        )
    if schema_only:
        rows_label = (
            f"~{n_rows_estimate:,} rows (estimate)"
            if n_rows_estimate is not None
            else "row count unknown"
        )
    else:
        rows_label = f"{n_rows_estimate:,} rows" if n_rows_estimate is not None else "—"
    console.print(
        Panel(
            head,
            title=f"[bold]Doctor[/bold] · {display_name} · {rows_label}",
            border_style="magenta",
            padding=(0, 1),
        )
    )

    targets = [n for n, s in stats.items() if s.kind in {"text", "bool"} and 2 <= s.n_unique <= 20]
    times = [n for n, s in stats.items() if s.kind == "temporal" and s.n_unique > 10]
    summary = []
    if targets:
        summary.append(f"target candidates: {', '.join(targets[:6])}")
    if times:
        summary.append(f"time candidates: {', '.join(times[:6])}")
    if not summary:
        summary.append("no obvious target/time candidates — pass --target/--time explicitly")
    console.print("[dim]" + " · ".join(summary) + "[/dim]")


def doctor_load(
    path: str, *, sample: int, credentials_env: str | None,
) -> tuple[dict[str, ColumnStats], str, int | None, bool]:
    """Resolve doctor input: warehouse schema discovery or sampled local load."""
    wh = discover_warehouse_schema(path, credentials_env=credentials_env)
    if wh is not None:
        stats, display, row_estimate = wh
        return stats, display, row_estimate, True
    src = cli.load(path, sample=sample, credentials_env=credentials_env)
    try:
        return cli.compute_all(src), src.source_name, src.n_rows, False
    finally:
        src.con.close()
