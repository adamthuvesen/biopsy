"""Typer-based CLI."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

import typer
from rich.console import Console

from sketch.demo import write_demo_csv
from sketch.profile import profile as profile_fn
from sketch.render.html import render as render_html
from sketch.render.terminal import render as render_terminal

app = typer.Typer(
    name="sketch",
    help="Instant, opinionated EDA in the terminal.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def profile(
    path: Path = typer.Argument(..., help="CSV, TSV, Parquet, or JSON file."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column for modeling."),
    time_col: str | None = typer.Option(
        None, "--time", help="Time column for temporal leakage analysis. Auto-detected if omitted."
    ),
    exclude: list[str] = typer.Option(
        [], "--exclude", "-x",
        help="Column to drop from analysis (target proxies, IDs). Repeatable: -x A -x B.",
    ),
    where: list[str] = typer.Option(
        [], "--filter", "-w",
        help="Filter expression. Repeatable. Examples: 'segment in train,test', 'value > 0'.",
    ),
    shortlist: int | None = typer.Option(
        None, "--shortlist", help="Cap the feature shortlist at N entries."
    ),
    cluster_cutoff: float = typer.Option(
        0.30, "--cluster-cutoff",
        help="Cluster cutoff on 1−|ρ| (default 0.30 ⇔ |ρ|≥0.70 collapses).",
    ),
    html: Path | None = typer.Option(None, "--html", help="Write an HTML supplement."),
    sample: int | None = typer.Option(None, "--sample", help="Sample N rows before profiling."),
    bins: int = typer.Option(24, "--bins", help="Histogram bin count."),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report in a browser."),
) -> None:
    """Profile a dataset and print a ranked report."""
    console = Console()
    prof = profile_fn(
        path, target=target, time_col=time_col, sample=sample, hist_bins=bins,
        cluster_cutoff=cluster_cutoff, shortlist_size=shortlist,
        exclude=list(exclude) if exclude else None,
        where=list(where) if where else None,
    )
    render_terminal(prof, console=console)

    if html or open_browser:
        out = html or Path(tempfile.gettempdir()) / f"sketch-{path.stem}.html"
        rendered = render_html(prof, out)
        console.print(f"\n[dim]HTML report:[/dim] {rendered}")
        if open_browser:
            webbrowser.open(rendered.as_uri())


@app.command()
def demo(
    n: int = typer.Option(5000, "--rows", help="Number of rows in synthetic dataset."),
    html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML supplement."),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report."),
) -> None:
    """Generate a synthetic dataset and profile it."""
    console = Console()
    tmpdir = Path(tempfile.mkdtemp(prefix="sketch-demo-"))
    csv_path = tmpdir / "demo.csv"
    write_demo_csv(csv_path, n=n)
    console.print(f"[dim]Generated:[/dim] {csv_path}")

    prof = profile_fn(csv_path, target="churned")
    render_terminal(prof, console=console)

    if html:
        out = tmpdir / "demo.html"
        rendered = render_html(prof, out)
        console.print(f"\n[dim]HTML report:[/dim] {rendered}")
        if open_browser:
            webbrowser.open(rendered.as_uri())


if __name__ == "__main__":
    app()
