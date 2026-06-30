"""biopsy demo and render saved profile."""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich.console import Console

import biopsy.cli as cli
from biopsy.cli.common import open_browser_if_requested, print_artifact_path
from biopsy.demo import write_demo_csv
from biopsy.profile import load_profile
from biopsy.render.html import render as render_html


def render_saved_profile_cmd(
    profile_json: Path = typer.Argument(..., help="Profile JSON from `biopsy profile --save`."),
    html: Path = typer.Option(..., "--html", help="HTML output path."),
    plotly_cdn: bool = typer.Option(
        False, "--plotly-cdn", help="Use Plotly from CDN instead of embedding it in HTML."
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report in a browser."),
) -> None:
    """Render a saved profile artifact without re-reading the dataset."""
    console = Console()
    prof = load_profile(profile_json)
    rendered = render_html(prof, html, embed_plotly=not plotly_cdn)
    print_artifact_path(console, "HTML report", rendered)
    open_browser_if_requested(rendered, enabled=open_browser)


def demo_cmd(
    n: int = typer.Option(5000, "--rows", min=1, help="Number of rows in synthetic dataset."),
    html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML supplement."),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report."),
) -> None:
    """Generate a synthetic dataset and profile it."""
    console = Console()
    tmpdir = Path(tempfile.mkdtemp(prefix="biopsy-demo-"))
    csv_path = tmpdir / "demo.csv"
    write_demo_csv(csv_path, n=n)
    console.print(f"[dim]demo.csv[/dim]  {n:,} rows", highlight=False)

    prof = cli.profile_fn(csv_path, target="churned")
    cli.render_terminal(prof, console=console)

    if html:
        out = tmpdir / "demo.html"
        rendered = render_html(prof, out)
        print_artifact_path(console, "HTML report", rendered, blank_before=True)
        open_browser_if_requested(rendered, enabled=open_browser)
