"""Shared CLI error handling and warehouse warnings."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

import typer
from rich.console import Console

from biopsy.warehouse import (
    MissingCredentialError,
    WarehouseDriverNotInstalledError,
)

USER_ERRORS: tuple[type[Exception], ...] = (
    MissingCredentialError,
    WarehouseDriverNotInstalledError,
    FileNotFoundError,
    ValueError,
    NotImplementedError,
)


def clean_exit_on_user_error(exc: Exception) -> None:
    """Print the message to stderr and exit non-zero — no traceback."""
    Console(stderr=True).print(f"[red]biopsy:[/red] {exc}")
    raise typer.Exit(code=2) from None


def write_text_artifact(path: str | Path, content: str) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def print_artifact_path(
    console: Console,
    label: str,
    path: Path,
    *,
    blank_before: bool = False,
) -> None:
    prefix = "\n" if blank_before else ""
    console.print(f"{prefix}[dim]{label}:[/dim] {path}")


def open_browser_if_requested(path: Path, *, enabled: bool) -> None:
    if enabled:
        webbrowser.open(path.as_uri())


def default_profile_html_path(source_name: str) -> Path:
    stem = Path(source_name).stem or "report"
    return Path(tempfile.gettempdir()) / f"biopsy-{stem}.html"


def default_compare_html_path(a_source_name: str, b_source_name: str) -> Path:
    a_stem = Path(a_source_name).stem or "a"
    b_stem = Path(b_source_name).stem or "b"
    return Path(tempfile.gettempdir()) / f"biopsy-compare-{a_stem}-{b_stem}.html"


def maybe_warn_warehouse_sample(
    progress_console: Console,
    path: str,
    sample: int | None,
) -> None:
    """Print a one-shot stderr warning if `--sample` is used against a URI."""
    if sample is None or "://" not in path:
        return
    from biopsy.warehouse import parse_warehouse_uri

    try:
        parsed = parse_warehouse_uri(path)
    except ValueError:
        return
    if parsed is None:
        return
    progress_console.print(
        "[yellow]biopsy:[/yellow] sample is head-of-table on warehouse sources "
        "(not random). Use --filter for stratification."
    )
