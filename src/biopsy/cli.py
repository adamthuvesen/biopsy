"""Typer-based CLI."""

from __future__ import annotations

import tempfile
import tomllib
import webbrowser
from math import isfinite
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from biopsy.demo import write_demo_csv
from biopsy.findings import _looks_like_id
from biopsy.io import load
from biopsy.profile import load_profile
from biopsy.profile import profile as profile_fn
from biopsy.render.html import render as render_html
from biopsy.render.terminal import render as render_terminal
from biopsy.stats import ColumnStats, compute_all
from biopsy.warehouse import (
    MissingCredentialError,
    WarehouseDriverNotInstalledError,
)

app = typer.Typer(
    name="biopsy",
    help="Instant, opinionated EDA in the terminal.",
    no_args_is_help=True,
    add_completion=False,
)


# Errors we treat as "user input is wrong" — show a clean one-line message
# and exit with code 2 rather than dumping a Rich traceback.
_USER_ERRORS: tuple[type[Exception], ...] = (
    MissingCredentialError,
    WarehouseDriverNotInstalledError,
    FileNotFoundError,
    ValueError,
    NotImplementedError,
)


def _clean_exit_on_user_error(exc: Exception) -> None:
    """Print the message to stderr and exit non-zero — no traceback."""
    Console(stderr=True).print(f"[red]biopsy:[/red] {exc}")
    raise typer.Exit(code=2) from None


@app.command()
def profile(
    path: str = typer.Argument(
        ...,
        help=(
            "Data source: file path (.csv/.tsv/.parquet/.json) or warehouse "
            "URI (s3://, https://, gs://, postgres://, snowflake://, bigquery://)."
        ),
    ),
    config: Path | None = typer.Option(
        None, "--config", help="TOML config with target, filters, exclusions, and output defaults."
    ),
    config_profile: str | None = typer.Option(
        None, "--profile-name", help="Profile name under [profiles.NAME] in --config."
    ),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column for modeling."),
    time_col: str | None = typer.Option(
        None, "--time", help="Time column for temporal leakage analysis. Auto-detected if omitted."
    ),
    exclude: list[str] = typer.Option(
        [], "--exclude", "-x",
        help="Column to drop from analysis (target proxies, IDs). Repeatable: -x A -x B.",
    ),
    exclude_file: Path | None = typer.Option(
        None, "--exclude-file", help="Text file with one excluded column per line."
    ),
    ignore_missing_exclude: bool | None = typer.Option(
        None, "--ignore-missing-exclude", help="Skip absent --exclude columns instead of failing."
    ),
    where: list[str] = typer.Option(
        [], "--filter", "-w",
        help="Filter expression. Repeatable. Examples: 'segment in train,test', 'value > 0'.",
    ),
    shortlist: int | None = typer.Option(
        None, "--shortlist", min=1, help="Cap the feature shortlist at N entries."
    ),
    cluster_cutoff: float | None = typer.Option(
        None, "--cluster-cutoff", min=0.0, max=1.0,
        help="Cluster cutoff on 1−|ρ| (default 0.30 ⇔ |ρ|≥0.70 collapses).",
    ),
    html: Path | None = typer.Option(None, "--html", help="Write an HTML supplement."),
    save: Path | None = typer.Option(None, "--save", help="Write the profile artifact as JSON."),
    pipeline: Path | None = typer.Option(
        None, "--pipeline",
        help="Write a runnable sklearn ColumnTransformer module from the action plan.",
    ),
    plotly_cdn: bool | None = typer.Option(
        None, "--plotly-cdn", help="Use Plotly from CDN instead of embedding it in HTML."
    ),
    sample: int | None = typer.Option(
        None, "--sample", min=1, help="Sample N rows before profiling."
    ),
    target_sample: int | None = typer.Option(
        None, "--target-sample", min=100,
        help="Rows to sample for target metrics; stratified for low-cardinality targets.",
    ),
    fast: bool | None = typer.Option(
        None, "--fast/--deep",
        help="Fast skips pairwise MI and target permutation. Use --deep for the full analysis.",
    ),
    all_columns: bool | None = typer.Option(
        None, "--all-columns", help="Print the full terminal column table."
    ),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Print major profiling phases to stderr."
    ),
    bins: int | None = typer.Option(None, "--bins", min=1, help="Histogram bin count."),
    max_cols: int | None = typer.Option(
        None, "--max-cols", min=2,
        help="Cap the number of columns in the pairwise MI pass. "
             "Useful on wide datasets to keep runtime sub-linear.",
    ),
    credentials_env: str | None = typer.Option(
        None, "--credentials-env",
        help=(
            "Prefix for warehouse credential env vars. With 'STAGING', biopsy "
            "reads STAGING_SNOWFLAKE_USER etc. instead of SNOWFLAKE_USER."
        ),
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report in a browser."),
) -> None:
    """Profile a dataset and print a ranked report."""
    cfg = _load_cli_config(config, config_profile)
    target = _coalesce(target, cfg.get("target"))
    time_col = _coalesce(time_col, cfg.get("time"), cfg.get("time_col"))
    shortlist = _coalesce(shortlist, cfg.get("shortlist"))
    cluster_cutoff = _coalesce(cluster_cutoff, cfg.get("cluster_cutoff"))
    html = _coalesce(html, _path_or_none(cfg.get("html")))
    save = _coalesce(save, _path_or_none(cfg.get("save")))
    sample = _coalesce(sample, cfg.get("sample"))
    bins = _coalesce(bins, cfg.get("bins"))
    target_sample = _coalesce(target_sample, cfg.get("target_sample"))
    fast = _coalesce(fast, _fast_from_config(cfg))
    max_cols = _coalesce(max_cols, cfg.get("max_cols"))
    all_columns = _coalesce(all_columns, cfg.get("all_columns"))
    plotly_cdn = _coalesce(plotly_cdn, cfg.get("plotly_cdn"))
    ignore_missing_exclude = _coalesce(
        ignore_missing_exclude, cfg.get("ignore_missing_exclude")
    )
    sample = _int_option("sample", sample, default=None, min_value=1)
    shortlist = _int_option("shortlist", shortlist, default=None, min_value=1)
    cluster_cutoff = _float_option(
        "cluster_cutoff", cluster_cutoff, default=0.30, min_value=0.0, max_value=1.0
    )
    bins = _int_option("bins", bins, default=24, min_value=1)
    target_sample = _int_option("target_sample", target_sample, default=30_000, min_value=100)
    max_cols = _int_option("max_cols", max_cols, default=None, min_value=2)
    fast = _bool_option("fast", fast, default=True)
    all_columns = _bool_option("all_columns", all_columns, default=False)
    plotly_cdn = _bool_option("plotly_cdn", plotly_cdn, default=False)
    ignore_missing_exclude = _bool_option(
        "ignore_missing_exclude", ignore_missing_exclude, default=False
    )

    cfg_exclude = _string_list(cfg.get("exclude"))
    file_exclude = _read_exclude_file(exclude_file or _path_or_none(cfg.get("exclude_file")))
    exclude_cols = [*cfg_exclude, *file_exclude, *exclude]
    where_filters = [*_string_list(cfg.get("filter")), *_string_list(cfg.get("where")), *where]

    console = Console(width=120)
    progress_console = Console(stderr=True, width=120)
    def show_progress(message: str) -> None:
        progress_console.print(f"[dim]biopsy:[/dim] {message}")

    # Warn once when sampling against a warehouse source — LIMIT N is
    # head-of-storage, not random, and can be biased.
    _maybe_warn_warehouse_sample(progress_console, path, sample)

    progress_cb = show_progress if progress else None
    try:
        prof = profile_fn(
            path, target=target, time_col=time_col, sample=sample, hist_bins=bins,
            cluster_cutoff=cluster_cutoff, shortlist_size=shortlist,
            exclude=exclude_cols or None,
            ignore_missing_exclude=ignore_missing_exclude,
            where=where_filters or None,
            deep_correlations=not fast,
            target_permutation=not fast,
            target_sample_size=target_sample,
            stratify_target=True,
            max_cols=max_cols,
            credentials_env=credentials_env,
            progress=progress_cb,
        )
    except _USER_ERRORS as exc:
        _clean_exit_on_user_error(exc)
    render_terminal(prof, console=console, all_columns=all_columns)

    if save:
        saved = prof.save(save)
        console.print(f"\n[dim]Profile JSON:[/dim] {saved}")

    if pipeline:
        pipeline_path = Path(pipeline).expanduser().resolve()
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_path.write_text(prof.to_sklearn_pipeline_code(), encoding="utf-8")
        console.print(f"\n[dim]Sklearn pipeline:[/dim] {pipeline_path}")

    if html or open_browser:
        # Derive a clean default filename from the source's display name.
        # source_name is `Path.name` for files and the last URI segment for
        # warehouse sources — `.stem` strips a trailing extension if any.
        stem = Path(prof.source_name).stem or "report"
        out = html if html is not None else (
            Path(tempfile.gettempdir()) / f"biopsy-{stem}.html"
        )
        rendered = render_html(prof, out, embed_plotly=not plotly_cdn)
        console.print(f"\n[dim]HTML report:[/dim] {rendered}")
        if open_browser:
            webbrowser.open(rendered.as_uri())


@app.command("render")
def render_saved_profile(
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
    console.print(f"[dim]HTML report:[/dim] {rendered}")
    if open_browser:
        webbrowser.open(rendered.as_uri())


@app.command()
def compare(
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

    _maybe_warn_warehouse_sample(progress_console, a, sample)
    _maybe_warn_warehouse_sample(progress_console, b, sample)

    try:
        prof_a = _load_side(
            a, "A", target, time_col, where, sample, credentials_env, show if progress else None
        )
        prof_b = _load_side(
            b, "B", target, time_col, where, sample, credentials_env, show if progress else None
        )
    except _USER_ERRORS as exc:
        _clean_exit_on_user_error(exc)
    report = compare_profiles(prof_a, prof_b)
    _print_compare(console, report)

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


@app.command()
def notebook(
    output: Path = typer.Argument(..., help="Notebook (.ipynb) path to write."),
    data_file: Path = typer.Option(..., "--file", help="Data file to profile."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column."),
    time_col: str | None = typer.Option(None, "--time", help="Time column."),
) -> None:
    """Write a starter notebook scaffolded against a profile's action plan."""
    console = Console()
    prof = profile_fn(data_file, target=target, time_col=time_col)
    pipeline_code = prof.to_sklearn_pipeline_code()
    plan = prof.action_plan()
    shortlist = (
        [e.feature for e in prof.clusters.shortlist[:20]] if prof.clusters else []
    )
    split_detail = plan.split.detail if plan.split else "Random 80/20 holdout"
    cv_detail = plan.cv.detail if plan.cv else "KFold(n_splits=5)"
    class_detail = plan.class_strategy.detail if plan.class_strategy else None

    notebook_json = _starter_notebook(
        data_file=data_file,
        target=target,
        time_col=time_col,
        pipeline_code=pipeline_code,
        shortlist=shortlist,
        split_detail=split_detail,
        cv_detail=cv_detail,
        class_detail=class_detail,
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(notebook_json, encoding="utf-8")
    console.print(f"[dim]Notebook:[/dim] {output}")


def _starter_notebook(
    *,
    data_file: Path,
    target: str | None,
    time_col: str | None,
    pipeline_code: str,
    shortlist: list[str],
    split_detail: str,
    cv_detail: str,
    class_detail: str | None,
) -> str:
    import json as _json

    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    def md_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    target_repr = repr(target) if target else "None"
    shortlist_repr = repr(shortlist)
    extras = []
    if class_detail:
        extras.append(f"# Class strategy: {class_detail}")
    extras_block = "\n".join(extras)

    cells = [
        md_cell(
            f"# Starter notebook — {data_file.name}\n\n"
            f"Generated by `biopsy notebook`. Target: `{target}`."
        ),
        code_cell(
            "import pandas as pd\n"
            "from sklearn.metrics import roc_auc_score, mean_absolute_error\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "from sklearn.pipeline import Pipeline\n"
        ),
        md_cell("## Load"),
        code_cell(
            f"DATA = {str(data_file)!r}\n"
            f"TARGET = {target_repr}\n"
            f"SHORTLIST = {shortlist_repr}\n"
            "if DATA.endswith(('.csv',)):\n"
            "    df = pd.read_csv(DATA)\n"
            "elif DATA.endswith(('.tsv', '.txt')):\n"
            "    df = pd.read_csv(DATA, sep='\\t')\n"
            "elif DATA.endswith(('.json',)):\n"
            "    df = pd.read_json(DATA)\n"
            "elif DATA.endswith(('.jsonl', '.ndjson')):\n"
            "    df = pd.read_json(DATA, lines=True)\n"
            "else:\n"
            "    df = pd.read_parquet(DATA)\n"
            "df.head()\n"
        ),
        md_cell("## Preprocessor (from biopsy action plan)"),
        code_cell(pipeline_code),
        md_cell(
            f"## Split\n\n{split_detail}\n\n## CV\n\n{cv_detail}\n"
            + (f"\n{extras_block}\n" if extras_block else "")
        ),
        code_cell(
            "from sklearn.model_selection import train_test_split\n\n"
            "X = df.drop(columns=[TARGET]) if TARGET else df\n"
            "y = df[TARGET] if TARGET else None\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y if y is not None else None\n"
            ")\n"
        ),
        md_cell("## Baseline model on the shortlist"),
        code_cell(
            "# `build_preprocessor` is defined by the cell above (biopsy codegen).\n"
            "assert 'build_preprocessor' in dir(), "
            "\"Pipeline cell did not define build_preprocessor()\"\n"
            "preproc = build_preprocessor()\n"
            "pipe = Pipeline([\n"
            "    ('preprocess', preproc),\n"
            "    ('model', GradientBoostingClassifier(random_state=42)),\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None\n"
            "if proba is not None:\n"
            "    print('AUC =', roc_auc_score(y_test, proba))\n"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return _json.dumps(notebook, indent=1)


@app.command()
def doctor(
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
        stats, display_name, n_rows_estimate, schema_only = _doctor_load(
            path, sample=sample, credentials_env=credentials_env,
        )
    except _USER_ERRORS as exc:
        _clean_exit_on_user_error(exc)
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
        looks: list[str] = []
        if _looks_like_id(s.name):
            looks.append("identifier")
        if s.kind == "numeric" and s.n_unique <= 2:
            looks.append("boolean")
        if s.kind in {"text", "bool"} and 2 <= s.n_unique <= 20:
            looks.append("low-card categorical")
        if s.kind == "numeric" and 2 < s.n_unique <= 20:
            looks.append("ordinal candidate target")
        if s.kind == "temporal":
            looks.append("time column candidate")
        if s.null_rate >= 0.5:
            looks.append("high-null")
        head.add_row(
            s.name, s.dtype.lower(), s.kind,
            f"{s.null_rate:.0%}" if s.null_rate else "—",
            f"{s.n_unique:,}",
            ", ".join(looks),
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


def _maybe_warn_warehouse_sample(
    progress_console: Console, path: str, sample: int | None,
) -> None:
    """Print a one-shot stderr warning if `--sample` is used against a URI.

    Warehouse sources translate `--sample N` to `LIMIT N`, which is
    head-of-storage (not random). Comparisons between samples can show
    false drift from sampling differences alone.
    """
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


def _doctor_load(
    path: str, *, sample: int, credentials_env: str | None,
) -> tuple[dict[str, ColumnStats], str, int | None, bool]:
    """Resolve doctor's input into (stats, display_name, n_rows, schema_only).

    For warehouse URIs, uses cheap schema-only discovery (Parquet footer
    for object stores; `information_schema` + `pg_class.reltuples` for
    Postgres). For paths and in-memory frames, falls through to
    `compute_all` on a sampled view (existing behavior).
    """
    import duckdb

    from biopsy.warehouse import parse_warehouse_uri

    parsed = parse_warehouse_uri(path)
    if parsed is not None and parsed.scheme in {"s3", "s3a", "https", "http", "gs", "gcs"}:
        from biopsy.warehouse.object_store import discover_schema as object_store_schema

        con = duckdb.connect(":memory:")
        try:
            schema = object_store_schema(con, parsed)
        finally:
            con.close()
        return _doctor_stats_from_schema(schema, None, parsed.qualified)

    if parsed is not None and parsed.scheme in {"postgres", "postgresql"}:
        from biopsy.warehouse.postgres import discover_schema as postgres_schema

        con = duckdb.connect(":memory:")
        try:
            schema, row_estimate = postgres_schema(
                con, parsed, credentials_env=credentials_env,
            )
        finally:
            con.close()
        return _doctor_stats_from_schema(schema, row_estimate, parsed.qualified)

    if parsed is not None and parsed.scheme == "bigquery":
        from biopsy.warehouse.bigquery import discover_schema as bq_schema

        schema, row_estimate = bq_schema(parsed, credentials_env=credentials_env)
        return _doctor_stats_from_schema(schema, row_estimate, parsed.qualified)

    if parsed is not None and parsed.scheme == "snowflake":
        from biopsy.warehouse.snowflake import discover_schema as sf_schema

        schema, row_estimate = sf_schema(parsed, credentials_env=credentials_env)
        return _doctor_stats_from_schema(schema, row_estimate, parsed.qualified)

    src = load(path, sample=sample, credentials_env=credentials_env)
    try:
        return compute_all(src), src.source_name, src.n_rows, False
    finally:
        src.con.close()


def _doctor_stats_from_schema(
    schema: dict[str, str], row_estimate: int | None, qualified: str,
) -> tuple[dict[str, ColumnStats], str, int | None, bool]:
    """Convert a schema-only discovery result into the doctor's tuple shape.

    Most ColumnStats fields are unknown (no rows pulled). The fields the
    doctor actually consults — `name`, `dtype`, `kind` — are populated;
    everything else defaults to zero.
    """
    from biopsy.io import kind_of as _kind_of

    stats: dict[str, ColumnStats] = {
        name: ColumnStats(
            name=name, dtype=dtype, kind=_kind_of(dtype),
            n=0, n_null=0, n_unique=0, null_rate=0.0,
        )
        for name, dtype in schema.items()
    }
    display = qualified.rsplit("/", 1)[-1] or qualified
    return stats, display, row_estimate, True


@app.command()
def diff(
    a: Path = typer.Argument(..., help="Side A: saved profile JSON."),
    b: Path = typer.Argument(..., help="Side B: saved profile JSON."),
) -> None:
    """Finding-level diff between two saved profiles."""
    console = Console(width=120)
    prof_a = load_profile(a)
    prof_b = load_profile(b)
    d = prof_a.diff(prof_b)
    _print_diff(console, d)


@app.command()
def demo(
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

    prof = profile_fn(csv_path, target="churned")
    render_terminal(prof, console=console)

    if html:
        out = tmpdir / "demo.html"
        rendered = render_html(prof, out)
        console.print(f"\n[dim]HTML report:[/dim] {rendered}")
        if open_browser:
            webbrowser.open(rendered.as_uri())


@app.command("init")
def init_config(
    path: Path = typer.Argument(..., help="Dataset to inspect for config defaults."),
    output: Path = typer.Option(
        Path("biopsy.toml"), "--output", "-o", help="Config file to write."
    ),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column to write."),
    time_col: str | None = typer.Option(None, "--time", help="Time column to write."),
    sample: int = typer.Option(10_000, "--sample", min=1, help="Rows to sample for inference."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing config file."),
) -> None:
    """Infer a starter biopsy.toml from a dataset."""
    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        raise typer.BadParameter(f"{output} already exists; pass --overwrite to replace it.")

    src = load(path, sample=sample)
    try:
        stats = compute_all(src)
    finally:
        src.con.close()
    if target is not None and target not in stats:
        raise typer.BadParameter(f"Target column '{target}' not found in {path}.")
    if time_col is not None and time_col not in stats:
        raise typer.BadParameter(f"Time column '{time_col}' not found in {path}.")

    inferred_target = target or _infer_target(stats)
    inferred_time = time_col or _infer_time(stats)
    excludes = _infer_excludes(stats, target=inferred_target, time_col=inferred_time)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _init_config_text(
            source=path,
            target=inferred_target,
            time_col=inferred_time,
            excludes=excludes,
        ),
        encoding="utf-8",
    )

    console = Console()
    console.print(f"[dim]Wrote:[/dim] {output}")
    if inferred_target:
        console.print(f"[dim]Target:[/dim] {inferred_target}")
    if inferred_time:
        console.print(f"[dim]Time:[/dim] {inferred_time}")
    if excludes:
        console.print(f"[dim]Excludes:[/dim] {', '.join(excludes[:12])}")


def _load_side(
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

    Accepts file paths, warehouse URIs, and saved profile JSONs. JSON
    detection is by suffix; URIs and paths flow through to `profile_fn`,
    which handles dispatch.
    """
    text = str(path)
    is_uri = "://" in text
    # Saved profiles are local files with .json suffix. Don't treat a
    # warehouse URI ending in .json (e.g. s3://bucket/x.json) as a saved
    # profile.
    if not is_uri and text.lower().endswith(".json"):
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

    return profile_fn(
        text,
        target=target,
        time_col=time_col,
        where=where or None,
        sample=sample,
        credentials_env=credentials_env,
        progress=cb if show else None,
    )


def _print_compare(console: Console, report: Any) -> None:
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


def _print_diff(console: Console, d: Any) -> None:
    from rich.panel import Panel
    from rich.table import Table

    console.print(Panel(
        f"[bold]Profile diff[/bold] · {d.a_name} → {d.b_name}",
        border_style="magenta", padding=(0, 2),
    ))
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
        console.print(Panel(t, title="[bold]Severity change[/bold]",
                            border_style="bright_black"))

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
        console.print(Panel(t, title="[bold]Target-signal rank change[/bold]",
                            border_style="bright_black"))


_CONFIG_KNOWN_KEYS: frozenset[str] = frozenset({
    "target", "time", "time_col", "exclude", "exclude_file", "ignore_missing_exclude",
    "filter", "where", "sample", "target_sample", "shortlist", "cluster_cutoff",
    "html", "save", "plotly_cdn", "fast", "deep", "all_columns", "bins", "max_cols",
})


def _load_cli_config(path: Path | None, profile_name: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    expanded = path.expanduser()
    try:
        text = expanded.read_text()
    except OSError as exc:
        raise typer.BadParameter(f"Cannot read config {expanded}: {exc}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise typer.BadParameter(f"Malformed TOML in {expanded}: {exc}") from exc
    profiles = data.get("profiles", {})
    cfg = {k: v for k, v in data.items() if k != "profiles"}
    _check_config_keys(cfg, path, where="top-level")
    if profile_name:
        selected = profiles.get(profile_name)
        if selected is None:
            raise typer.BadParameter(f"Profile '{profile_name}' not found in {path}.")
        _check_config_keys(selected, path, where=f"[profiles.{profile_name}]")
        cfg.update(selected)
    return cfg


def _check_config_keys(cfg: dict[str, Any], path: Path, *, where: str) -> None:
    import difflib

    unknown = sorted(set(cfg) - _CONFIG_KNOWN_KEYS)
    if not unknown:
        return
    suggestions = []
    for k in unknown:
        match = difflib.get_close_matches(k, sorted(_CONFIG_KNOWN_KEYS), n=1, cutoff=0.6)
        if match:
            suggestions.append(f"'{k}' (did you mean '{match[0]}'?)")
        else:
            suggestions.append(f"'{k}'")
    raise typer.BadParameter(
        f"Unknown config key(s) in {path} {where}: {', '.join(suggestions)}."
    )


def _fast_from_config(cfg: dict[str, Any]) -> Any:
    """Resolve TOML `fast` / `deep` aliases into the internal fast flag."""
    fast = _bool_config_value(cfg, "fast")
    deep = _bool_config_value(cfg, "deep")
    has_fast = fast is not None
    has_deep = deep is not None
    if has_fast and has_deep:
        if fast == deep:
            raise typer.BadParameter(
                "Config keys 'fast' and 'deep' conflict. Use one key, or set "
                "`fast = false` with `deep = true`."
            )
        return fast
    if has_fast:
        return cfg["fast"]
    if has_deep:
        return not bool(cfg["deep"])
    return None


def _bool_config_value(cfg: dict[str, Any], key: str) -> bool | None:
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise typer.BadParameter(f"Config key '{key}' must be true or false.")
    return value


def _bool_option(name: str, value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise typer.BadParameter(f"Config key '{name}' must be true or false.")
    return value


def _int_option(
    name: str,
    value: Any,
    *,
    default: int | None,
    min_value: int | None = None,
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise typer.BadParameter(f"Config key '{name}' must be an integer.")
    if min_value is not None and value < min_value:
        raise typer.BadParameter(f"Config key '{name}' must be >= {min_value}.")
    return value


def _float_option(
    name: str,
    value: Any,
    *,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise typer.BadParameter(f"Config key '{name}' must be a number.")
    coerced = float(value)
    if not isfinite(coerced):
        raise typer.BadParameter(f"Config key '{name}' must be finite.")
    if min_value is not None and coerced < min_value:
        raise typer.BadParameter(f"Config key '{name}' must be >= {min_value}.")
    if max_value is not None and coerced > max_value:
        raise typer.BadParameter(f"Config key '{name}' must be <= {max_value}.")
    return coerced


def _coalesce(cli_value: Any, config_value: Any, *aliases: Any) -> Any:
    if config_value is None:
        for alias in aliases:
            if alias is not None:
                config_value = alias
                break
    return cli_value if cli_value not in (None, [], {}) else config_value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _read_exclude_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    expanded = path.expanduser()
    try:
        text = expanded.read_text()
    except OSError as exc:
        raise typer.BadParameter(f"Cannot read exclude file {expanded}: {exc}") from exc
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out


def _infer_target(stats: dict[str, ColumnStats]) -> str | None:
    exact = [
        "target",
        "label",
        "y",
        "outcome",
        "churn",
        "churned",
        "converted",
        "conversion",
        "default",
        "fraud",
        "is_fraud",
    ]
    by_lower = {name.lower(): name for name in stats}
    for candidate in exact:
        if candidate in by_lower:
            return by_lower[candidate]
    for name in stats:
        lower = name.lower()
        if lower.endswith("_target") or lower.startswith("target_"):
            return name
    for name, col_stats in stats.items():
        lower = name.lower()
        if lower.startswith(("is_", "has_")) and col_stats.n_unique == 2:
            return name
    return None


def _infer_time(stats: dict[str, ColumnStats]) -> str | None:
    temporal = [
        name for name, col_stats in stats.items()
        if col_stats.kind == "temporal" and col_stats.n_unique >= 10
    ]
    if len(temporal) == 1:
        return temporal[0]
    preferred_tokens = ("date", "time", "created", "event", "snapshot", "as_of")
    for name in temporal:
        lower = name.lower()
        if any(token in lower for token in preferred_tokens):
            return name
    return None


def _infer_excludes(
    stats: dict[str, ColumnStats],
    *,
    target: str | None,
    time_col: str | None,
) -> list[str]:
    keep = {name for name in (target, time_col) if name}
    excludes: list[str] = []
    for name, col_stats in stats.items():
        if name in keep:
            continue
        non_null = col_stats.n - col_stats.n_null
        if non_null <= 50:
            continue
        if _looks_like_id(name) and col_stats.unique_rate >= 0.95:
            excludes.append(name)
            continue
        if col_stats.kind == "text" and col_stats.unique_rate >= 0.80 and col_stats.n_unique > 100:
            excludes.append(name)
    return excludes


def _init_config_text(
    *,
    source: Path,
    target: str | None,
    time_col: str | None,
    excludes: list[str],
) -> str:
    lines = [
        "# Generated by `biopsy init`.",
        f"# Source inspected: {source.expanduser()}",
        "",
    ]
    if target:
        lines.append(f"target = {_toml_string(target)}")
    else:
        lines.append("# target = \"your_target_column\"")
    if time_col:
        lines.append(f"time = {_toml_string(time_col)}")
    else:
        lines.append("# time = \"your_time_column\"")
    lines.extend([
        f"exclude = {_toml_string_list(excludes)}",
        "filter = []",
        "fast = true",
        "target_sample = 30000",
        "ignore_missing_exclude = true",
        "plotly_cdn = true",
        "",
        "[profiles.deep]",
        "fast = false",
        "target_sample = 50000",
    ])
    return "\n".join(lines) + "\n"


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_string_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


if __name__ == "__main__":
    app()
