# Architecture

Annotated module map for `biopsy`. `AGENTS.md` carries the subpackage-level
tree; this is the per-file detail.

## `src/biopsy/`

Core profiling and analysis modules:

- `io.py` — DuckDB loader, `--filter` parser, `--exclude`. Owns identifier
  quoting (`_quote_ident`) and literal binding (`_lit`).
- `stats.py` — per-column SQL aggregates, histograms, batched base counts.
  Owns `_quote` for identifiers.
- `correlations.py` — Pearson, MI, PPS, Spearman, AUC, permutation importance,
  bootstrap CIs.
- `clustering.py` — Spearman-distance hierarchical clustering and shortlist.
- `temporal.py` — temporal leakage, drift, monotonicity, target drift.
- `findings.py` — ranked findings synthesis plus smart detectors.
- `action_plan.py` — drop / impute / encode / transform / split / cv decisions
  plus sklearn codegen. Single source of truth (see gotchas).
- `compare.py` — profile-to-profile drift (KS / Wasserstein / PSI / χ² / JS).
- `targets.py` — target column typing shared across profiling, correlations,
  and temporal.
- `inference.py` — heuristics for target/time candidates, ID detection, doctor
  hints.
- `matrix.py` — shared sampled matrices for one profile run.
- `serialize.py` — JSON-safe serialization helpers for public report objects.
- `demo.py` — synthetic dataset generator.
- `sparkline.py` — unicode sparklines for the terminal.

## `src/biopsy/profile/`

Orchestrator and public API (formerly `profile.py`):

- `run.py` — profiling pipeline orchestrator.
- `model.py` — `Profile` / `ProfileDiff` dataclasses.
- `diff.py` — finding-level diff between profiles.
- `serde.py` — profile <-> JSON load/save.

## `src/biopsy/cli/`

Typer CLI (formerly `cli.py`), one module per command:

- `profile_cmd.py`, `compare_cmd.py`, `diff_cmd.py`, `doctor_cmd.py`,
  `init_cmd.py`, `demo_cmd.py`, `notebook.py` — the commands.
- `config.py` — `biopsy.toml` parsing plus `CONFIG_KNOWN_KEYS` (keep aligned
  with CLI flags).
- `common.py` — shared option types and credential plumbing.

## `src/biopsy/warehouse/`

Remote sources: postgres, bigquery, snowflake (plus `object_store` and
`doctor`).

## `src/biopsy/render/`

- `terminal.py` — Rich terminal report (100-char cap layout).
- `charts.py` — Plotly chart builders.
- `html.py` — Plotly + Jinja HTML report and compare report.

## `src/biopsy/templates/`

- `report.html.j2`, `compare.html.j2` — Jinja templates for HTML output.

## `tests/`

Split from the old `test_smoke.py` into domain modules:

- `test_smoke.py` — core-path smoke plus regression tests for fixed bugs.
- `test_temporal.py`, `test_correlations.py`, `test_action_plan.py`,
  `test_findings_quality.py`, `test_compare.py`, `test_render.py`,
  `test_cli.py`, `test_profile_pipeline.py`, `test_filter_and_io.py` — domain
  suites.
- `test_warehouse*.py` — postgres (docker-compose), bigquery, snowflake,
  readonly.
