"""Typer-based CLI."""

from __future__ import annotations

import typer

from biopsy.cli.compare_cmd import compare_cmd
from biopsy.cli.compare_cmd import print_compare as _print_compare
from biopsy.cli.demo_cmd import demo_cmd, render_saved_profile_cmd
from biopsy.cli.diff_cmd import diff_cmd
from biopsy.cli.doctor_cmd import doctor_cmd
from biopsy.cli.doctor_cmd import doctor_load as _doctor_load
from biopsy.cli.init_cmd import init_config_cmd
from biopsy.cli.notebook import notebook_cmd
from biopsy.cli.notebook import starter_notebook as _starter_notebook
from biopsy.cli.profile_cmd import profile_cmd

# Re-exported for tests and commands that monkeypatch `biopsy.cli.load`, etc.
from biopsy.io import load
from biopsy.profile import profile as profile_fn
from biopsy.render.terminal import render as render_terminal
from biopsy.stats import compute_all

app = typer.Typer(
    name="biopsy",
    help="Instant, opinionated EDA in the terminal.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("profile")(profile_cmd)
app.command("render")(render_saved_profile_cmd)
app.command("compare")(compare_cmd)
app.command("notebook")(notebook_cmd)
app.command("doctor")(doctor_cmd)
app.command("diff")(diff_cmd)
app.command("demo")(demo_cmd)
app.command("init")(init_config_cmd)

__all__ = [
    "_doctor_load",
    "_print_compare",
    "_starter_notebook",
    "app",
    "compute_all",
    "load",
    "profile_fn",
    "render_terminal",
]


if __name__ == "__main__":
    app()
