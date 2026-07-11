# AGENTS.md: biopsy

`biopsy` is an opinionated EDA library + CLI. Point it at a CSV or Parquet and
get a ranked report of what matters for modeling: distributions, nulls,
outliers, non-linear correlations, target signal, temporal leakage, redundancy
clustering, drift comparison, and a runnable sklearn preprocessor + split/CV
recommendation. Differentiator vs `ydata-profiling` / `SweetViz` / `DataPrep`:
**ranking + opinion + action**.

User-level guidance (tone, principles, git etiquette) lives in
`~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated
here. This file is for project-specific facts.

## Layout

```
src/biopsy/
├── io.py            # DuckDB loader, --filter parser, --exclude
├── stats.py         # per-column SQL aggregates, histograms
├── correlations.py  # Pearson, MI, PPS, Spearman, AUC, perm importance
├── clustering.py    # Spearman-distance clustering + shortlist
├── temporal.py      # leakage, drift, monotonicity, target drift
├── findings.py      # ranked findings synthesis + detectors
├── action_plan.py   # drop/impute/encode/transform/split/cv + sklearn codegen
├── compare.py       # profile-to-profile drift
├── profile/         # orchestrator + public API (Profile dataclasses, serde)
├── cli/             # Typer CLI, one module per command
├── warehouse/       # remote sources: postgres, bigquery, snowflake
├── render/          # terminal (Rich), HTML with Plotly charts (Jinja)
└── templates/       # report.html.j2, compare.html.j2
tests/               # domain-split suites + test_warehouse*.py
```

Per-file detail is in [docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # dev environment

uv run python -m pytest tests/ -q   # test suite
uv run ruff check .                 # lint
uv run ruff format --check .        # format gate

biopsy demo --rows 5000             # synthetic dataset
biopsy profile data.parquet --target y
biopsy compare a.parquet b.parquet  # drift
biopsy diff a.json b.json           # finding-level diff
biopsy doctor data.parquet          # quick schema + candidates
```

## Critical Conventions

- **Python 3.11+, `uv`-managed.** Source under `src/`, package name `biopsy`.
- **DuckDB-first.** Column stats and Pearson correlations are computed in SQL.
  Use `_quote_ident(col)` (in [src/biopsy/io.py](src/biopsy/io.py)) and
  `_quote(col)` (in [src/biopsy/stats.py](src/biopsy/stats.py)) for identifiers;
  `_lit(value, is_numeric)` for literals. Never string-interpolate.
- **Library split.** sklearn for MI / PPS / AUC / permutation importance /
  bootstrap CIs; scipy for `spearmanr`, `ks_2samp`, `chi2_contingency`;
  plotly + jinja2 for HTML; rich + typer for CLI.
- **No blanket `except Exception:`** unless the failure is genuinely recoverable
  and documented. Silent failures hid a real bug in the v0.1 review.
- **Severity vocabulary:** `critical` / `warning` / `info`. Categories:
  `leakage`, `suspicious`, `quality`, `distribution`, `correlation`, `target`,
  `temporal`, `drift`.
- **`biopsy.toml` keys** in [src/biopsy/cli/config.py](src/biopsy/cli/config.py)
  (`CONFIG_KNOWN_KEYS`) must stay aligned with CLI flags.
- **Never commit secrets, `.env`, or AI-attribution lines.**

## Read The Docs First

Before changing a subsystem, read the matching doc:

- **Module architecture / per-file map** → [docs/architecture.md](docs/architecture.md)
- **Non-obvious algorithm + ordering gotchas** (filter parser, `USING SAMPLE`,
  Spearman tie-correction, perm-importance index alignment, leakage
  classification, action-plan SoT, …) → [docs/gotchas.md](docs/gotchas.md)

If a doc disagrees with code, fix the doc in the same change.

## Index

Start in [docs/architecture.md](docs/architecture.md), then read
[docs/gotchas.md](docs/gotchas.md) before touching any algorithm.
