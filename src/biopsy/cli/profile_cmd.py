"""biopsy profile — main profiling command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

import biopsy.cli as cli
from biopsy.cli.common import (
    USER_ERRORS,
    clean_exit_on_user_error,
    default_profile_html_path,
    maybe_warn_warehouse_sample,
    open_browser_if_requested,
    print_artifact_path,
    write_text_artifact,
)
from biopsy.cli.config import (
    bool_option,
    coalesce,
    fast_from_config,
    float_option,
    int_option,
    load_cli_config,
    path_or_none,
    read_exclude_file,
    string_list,
)
from biopsy.render.html import render as render_html


def profile_cmd(
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
        [],
        "--exclude",
        "-x",
        help="Column to drop from analysis (target proxies, IDs). Repeatable: -x A -x B.",
    ),
    exclude_file: Path | None = typer.Option(
        None, "--exclude-file", help="Text file with one excluded column per line."
    ),
    ignore_missing_exclude: bool | None = typer.Option(
        None, "--ignore-missing-exclude", help="Skip absent --exclude columns instead of failing."
    ),
    where: list[str] = typer.Option(
        [],
        "--filter",
        "-w",
        help="Filter expression. Repeatable. Examples: 'segment in train,test', 'value > 0'.",
    ),
    shortlist: int | None = typer.Option(
        None, "--shortlist", min=1, help="Cap the feature shortlist at N entries."
    ),
    cluster_cutoff: float | None = typer.Option(
        None,
        "--cluster-cutoff",
        min=0.0,
        max=1.0,
        help="Cluster cutoff on 1−|ρ| (default 0.30 ⇔ |ρ|≥0.70 collapses).",
    ),
    html: Path | None = typer.Option(None, "--html", help="Write an HTML supplement."),
    save: Path | None = typer.Option(None, "--save", help="Write the profile artifact as JSON."),
    pipeline: Path | None = typer.Option(
        None,
        "--pipeline",
        help="Write a runnable sklearn ColumnTransformer module from the action plan.",
    ),
    plotly_cdn: bool | None = typer.Option(
        None, "--plotly-cdn", help="Use Plotly from CDN instead of embedding it in HTML."
    ),
    sample: int | None = typer.Option(
        None, "--sample", min=1, help="Sample N rows before profiling."
    ),
    target_sample: int | None = typer.Option(
        None,
        "--target-sample",
        min=100,
        help="Rows to sample for target metrics; stratified for low-cardinality targets.",
    ),
    fast: bool | None = typer.Option(
        None,
        "--fast/--deep",
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
        None,
        "--max-cols",
        min=2,
        help="Cap the number of columns in the pairwise MI pass. "
        "Useful on wide datasets to keep runtime sub-linear.",
    ),
    credentials_env: str | None = typer.Option(
        None,
        "--credentials-env",
        help=(
            "Prefix for warehouse credential env vars. With 'STAGING', biopsy "
            "reads STAGING_SNOWFLAKE_USER etc. instead of SNOWFLAKE_USER."
        ),
    ),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report in a browser."),
) -> None:
    """Profile a dataset and print a ranked report."""
    cfg = load_cli_config(config, config_profile)
    target = coalesce(target, cfg.get("target"))
    time_col = coalesce(time_col, cfg.get("time"), cfg.get("time_col"))
    shortlist = coalesce(shortlist, cfg.get("shortlist"))
    cluster_cutoff = coalesce(cluster_cutoff, cfg.get("cluster_cutoff"))
    html = coalesce(html, path_or_none(cfg.get("html")))
    save = coalesce(save, path_or_none(cfg.get("save")))
    sample = coalesce(sample, cfg.get("sample"))
    bins = coalesce(bins, cfg.get("bins"))
    target_sample = coalesce(target_sample, cfg.get("target_sample"))
    fast = coalesce(fast, fast_from_config(cfg))
    max_cols = coalesce(max_cols, cfg.get("max_cols"))
    all_columns = coalesce(all_columns, cfg.get("all_columns"))
    plotly_cdn = coalesce(plotly_cdn, cfg.get("plotly_cdn"))
    ignore_missing_exclude = coalesce(ignore_missing_exclude, cfg.get("ignore_missing_exclude"))
    sample = int_option("sample", sample, default=None, min_value=1)
    shortlist = int_option("shortlist", shortlist, default=None, min_value=1)
    cluster_cutoff = float_option(
        "cluster_cutoff", cluster_cutoff, default=0.30, min_value=0.0, max_value=1.0
    )
    bins = int_option("bins", bins, default=24, min_value=1)
    target_sample = int_option("target_sample", target_sample, default=30_000, min_value=100)
    max_cols = int_option("max_cols", max_cols, default=None, min_value=2)
    fast = bool_option("fast", fast, default=True)
    all_columns = bool_option("all_columns", all_columns, default=False)
    plotly_cdn = bool_option("plotly_cdn", plotly_cdn, default=False)
    ignore_missing_exclude = bool_option(
        "ignore_missing_exclude", ignore_missing_exclude, default=False
    )

    cfg_exclude = string_list(cfg.get("exclude"))
    file_exclude = read_exclude_file(exclude_file or path_or_none(cfg.get("exclude_file")))
    exclude_cols = [*cfg_exclude, *file_exclude, *exclude]
    where_filters = [*string_list(cfg.get("filter")), *string_list(cfg.get("where")), *where]

    console = Console(width=120)
    progress_console = Console(stderr=True, width=120)

    def show_progress(message: str) -> None:
        progress_console.print(f"[dim]biopsy:[/dim] {message}")

    # Warn once when sampling against a warehouse source — LIMIT N is
    # head-of-storage, not random, and can be biased.
    maybe_warn_warehouse_sample(progress_console, path, sample)

    progress_cb = show_progress if progress else None
    try:
        prof = cli.profile_fn(
            path,
            target=target,
            time_col=time_col,
            sample=sample,
            hist_bins=bins,
            cluster_cutoff=cluster_cutoff,
            shortlist_size=shortlist,
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
    except USER_ERRORS as exc:
        clean_exit_on_user_error(exc)
    cli.render_terminal(prof, console=console, all_columns=all_columns)

    if save:
        saved = prof.save(save)
        print_artifact_path(console, "Profile JSON", saved, blank_before=True)

    if pipeline:
        pipeline_path = write_text_artifact(pipeline, prof.to_sklearn_pipeline_code())
        print_artifact_path(console, "Sklearn pipeline", pipeline_path, blank_before=True)

    if html or open_browser:
        out = html if html is not None else default_profile_html_path(prof.source_name)
        rendered = render_html(prof, out, embed_plotly=not plotly_cdn)
        print_artifact_path(console, "HTML report", rendered, blank_before=True)
        open_browser_if_requested(rendered, enabled=open_browser)
