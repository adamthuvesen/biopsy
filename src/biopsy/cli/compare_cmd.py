"""biopsy compare — drift between datasets or saved profiles."""

from __future__ import annotations

import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import biopsy.cli as cli
from biopsy.cli.common import USER_ERRORS, clean_exit_on_user_error, maybe_warn_warehouse_sample
from biopsy.profile import load_profile


def compare_cmd(
    a: str = typer.Argument(..., help="Side A: data file, warehouse URI, or profile JSON."),
    b: str = typer.Argument(..., help="Side B: data file, warehouse URI, or profile JSON."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target column (used when A and B are data files)."
    ),
    time_col: str | None = typer.Option(
        None, "--time", help="Time column (used when A and B are data files)."
    ),
    where: list[str] = typer.Option(
        [], "--filter", "-w", help="Filter applied to both sides when reading data files."
    ),
    sample: int | None = typer.Option(
        None, "--sample", min=1, help="Reservoir sample N rows per side."
    ),
    credentials_env: str | None = typer.Option(
        None,
        "--credentials-env",
        help="Prefix for warehouse credential env vars. See `biopsy profile --help`.",
    ),
    html: Path | None = typer.Option(None, "--html", help="Write a compare HTML report."),
    save: Path | None = typer.Option(None, "--save", help="Save the compare report JSON."),
    plotly_cdn: bool = typer.Option(
        False, "--plotly-cdn", help="Use Plotly from CDN instead of embedding."
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report."),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Print profiling phases."
    ),
) -> None:
    """Drift report comparing two datasets (or two saved profiles)."""
    from biopsy.compare import compare_profiles
    from biopsy.render.html import render_compare

    console = Console(width=120)
    progress_console = Console(stderr=True, width=120)

    def show(side: str, msg: str) -> None:
        progress_console.print(f"[dim]biopsy compare ({side}):[/dim] {msg}")

    maybe_warn_warehouse_sample(progress_console, a, sample)
    maybe_warn_warehouse_sample(progress_console, b, sample)

    try:
        prof_a = load_side(
            a, "A", target, time_col, where, sample, credentials_env, show if progress else None
        )
        prof_b = load_side(
            b, "B", target, time_col, where, sample, credentials_env, show if progress else None
        )
    except USER_ERRORS as exc:
        clean_exit_on_user_error(exc)
    report = compare_profiles(prof_a, prof_b)
    cli._print_compare(console, report)

    if save:
        import json as _json

        from biopsy.serialize import to_jsonable

        payload = {
            "a_name": report.a_name,
            "b_name": report.b_name,
            "schema": to_jsonable(report.schema),
            "drifts": [to_jsonable(d) for d in report.drifts],
            "target": to_jsonable(report.target) if report.target else None,
            "findings": [to_jsonable(f) for f in report.findings],
        }
        save_path = Path(save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(_json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"\n[dim]Compare JSON:[/dim] {save_path}")

    if html or open_browser:
        a_stem = Path(prof_a.source_name).stem or "a"
        b_stem = Path(prof_b.source_name).stem or "b"
        out = html if html is not None else (
            Path(tempfile.gettempdir()) / f"biopsy-compare-{a_stem}-{b_stem}.html"
        )
        rendered = render_compare(prof_a, prof_b, report, out, embed_plotly=not plotly_cdn)
        console.print(f"\n[dim]HTML report:[/dim] {rendered}")
        if open_browser:
            webbrowser.open(rendered.as_uri())


def load_side(
    path: str,
    label: str,
    target: str | None,
    time_col: str | None,
    where: list[str],
    sample: int | None,
    credentials_env: str | None,
    show: Any,
) -> Any:
    """Either load a saved profile JSON or profile a data file / URI in place.

    Accepts file paths, warehouse URIs, and saved profile JSONs. Local JSON
    files are treated as saved profiles only when they have biopsy's profile
    artifact shape; other JSON files flow through to `profile_fn` as data.
    """
    text = str(path)
    is_uri = "://" in text
    # Saved profiles are local files with .json suffix. Don't treat a
    # warehouse URI ending in .json (e.g. s3://bucket/x.json) as a saved
    # profile.
    if not is_uri and text.lower().endswith(".json") and looks_like_profile_json(Path(text)):
        display = Path(text).name
        if show:
            show(label, f"loading profile {display}")
        return load_profile(Path(text))

    display = text if is_uri else Path(text).name
    if show:
        show(label, f"profiling {display}")

    def cb(message: str) -> None:
        if show:
            show(label, message)

    return cli.profile_fn(
        text,
        target=target,
        time_col=time_col,
        where=where or None,
        sample=sample,
        credentials_env=credentials_env,
        progress=cb if show else None,
    )


def looks_like_profile_json(path: Path) -> bool:
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    required = {"source_name", "n_rows", "n_cols", "columns", "findings"}
    return required.issubset(payload)


def print_compare(console: Console, report: Any) -> None:
    """Curated terminal compare summary: schema diff, target delta,
    top-10 drifted columns."""
    from rich.panel import Panel
    from rich.table import Table

    console.print(
        Panel(
            f"[bold]Compare[/bold] · {report.a_name} → {report.b_name}",
            border_style="magenta",
            padding=(0, 2),
        )
    )

    schema = report.schema
    if schema.added or schema.removed or schema.type_changed:
        t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
        t.add_column("change", style="cyan")
        t.add_column("columns", overflow="fold")
        if schema.added:
            t.add_row("added", ", ".join(schema.added))
        if schema.removed:
            t.add_row("removed", ", ".join(schema.removed))
        if schema.type_changed:
            t.add_row(
                "type changed",
                ", ".join(f"{c} ({a}→{b})" for c, a, b in schema.type_changed),
            )
        console.print(Panel(t, title="[bold]Schema diff[/bold]", border_style="bright_black"))
    else:
        console.print("[dim]Schema unchanged.[/dim]")

    if report.target is not None:
        console.print(
            Panel(
                f"[bold]Target Δ[/bold] · {report.target.detail}",
                border_style="yellow",
                padding=(0, 2),
            )
        )

    t = Table(show_header=True, header_style="bold", border_style="bright_black", expand=True)
    t.add_column("column", style="cyan")
    t.add_column("kind", style="dim")
    t.add_column("KS", justify="right")
    t.add_column("PSI", justify="right")
    t.add_column("JS", justify="right")
    t.add_column("null Δ", justify="right")
    t.add_column("score", justify="right")
    for d in report.top(10):
        if d.drift_score <= 0:
            continue
        t.add_row(
            d.column,
            d.kind,
            f"{d.ks_stat:.2f}" if d.ks_stat is not None else "—",
            f"{d.psi:.2f}" if d.psi is not None else "—",
            f"{d.js_divergence:.2f}" if d.js_divergence is not None else "—",
            f"{d.null_rate_delta:+.0%}" if d.null_rate_delta is not None else "—",
            f"{d.drift_score:.2f}",
        )
    console.print(Panel(t, title="[bold]Top drift[/bold]", border_style="bright_black"))

    if report.findings:
        f_table = Table(show_header=True, header_style="bold", border_style="bright_black",
                        expand=True)
        f_table.add_column("severity", style="dim")
        f_table.add_column("title")
        f_table.add_column("detail", overflow="fold")
        for f in report.findings[:12]:
            color = {"critical": "red", "warning": "yellow", "info": "cyan"}[f.severity]
            f_table.add_row(f"[{color}]{f.severity}[/{color}]", f.title, f.detail)
        console.print(Panel(f_table, title="[bold]Drift findings[/bold]",
                            border_style="bright_black"))
