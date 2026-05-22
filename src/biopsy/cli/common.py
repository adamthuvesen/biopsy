"""Shared CLI error handling and warehouse warnings."""

from __future__ import annotations

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


def maybe_warn_warehouse_sample(
    progress_console: Console, path: str, sample: int | None,
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
