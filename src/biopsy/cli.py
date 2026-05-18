"""Typer-based CLI."""

from __future__ import annotations

import tempfile
import tomllib
import webbrowser
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

app = typer.Typer(
    name="biopsy",
    help="Instant, opinionated EDA in the terminal.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def profile(
    path: Path = typer.Argument(..., help="CSV, TSV, Parquet, or JSON file."),
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
    fast = _coalesce(fast, cfg.get("fast"))
    all_columns = _coalesce(all_columns, cfg.get("all_columns"))
    plotly_cdn = _coalesce(plotly_cdn, cfg.get("plotly_cdn"))
    ignore_missing_exclude = _coalesce(
        ignore_missing_exclude, cfg.get("ignore_missing_exclude")
    )
    cluster_cutoff = 0.30 if cluster_cutoff is None else float(cluster_cutoff)
    bins = 24 if bins is None else int(bins)
    target_sample = 30_000 if target_sample is None else int(target_sample)
    fast = True if fast is None else bool(fast)
    all_columns = False if all_columns is None else bool(all_columns)
    plotly_cdn = False if plotly_cdn is None else bool(plotly_cdn)
    ignore_missing_exclude = (
        False if ignore_missing_exclude is None else bool(ignore_missing_exclude)
    )

    cfg_exclude = _string_list(cfg.get("exclude"))
    file_exclude = _read_exclude_file(exclude_file or _path_or_none(cfg.get("exclude_file")))
    exclude_cols = [*cfg_exclude, *file_exclude, *exclude]
    where_filters = [*_string_list(cfg.get("filter")), *_string_list(cfg.get("where")), *where]

    console = Console(width=120)
    progress_console = Console(stderr=True, width=120)
    def show_progress(message: str) -> None:
        progress_console.print(f"[dim]biopsy:[/dim] {message}")

    progress_cb = show_progress if progress else None
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
        progress=progress_cb,
    )
    render_terminal(prof, console=console, all_columns=all_columns)

    if save:
        saved = prof.save(save)
        console.print(f"\n[dim]Profile JSON:[/dim] {saved}")

    if html or open_browser:
        out = html or Path(tempfile.gettempdir()) / f"biopsy-{path.stem}.html"
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
def demo(
    n: int = typer.Option(5000, "--rows", help="Number of rows in synthetic dataset."),
    html: bool = typer.Option(True, "--html/--no-html", help="Generate HTML supplement."),
    open_browser: bool = typer.Option(False, "--open", help="Open the HTML report."),
) -> None:
    """Generate a synthetic dataset and profile it."""
    console = Console()
    tmpdir = Path(tempfile.mkdtemp(prefix="biopsy-demo-"))
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
    stats = compute_all(src)
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


def _load_cli_config(path: Path | None, profile_name: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = tomllib.loads(path.expanduser().read_text())
    cfg = {k: v for k, v in data.items() if k != "profiles"}
    profiles = data.get("profiles", {})
    if profile_name:
        selected = profiles.get(profile_name)
        if selected is None:
            raise typer.BadParameter(f"Profile '{profile_name}' not found in {path}.")
        cfg.update(selected)
    return cfg


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
    out: list[str] = []
    for line in path.expanduser().read_text().splitlines():
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
