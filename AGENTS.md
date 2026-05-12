# AGENTS.md — biopsy

Project-specific instructions for AI coding agents working on this repo. The user's global `~/dotfiles/agents/AGENTS.md` applies on top — this file overrides for biopsy-specific concerns.

## What this is

`biopsy` is an opinionated EDA library + CLI. Point it at a CSV or Parquet, get a ranked report of what actually matters for modeling — distributions, nulls, outliers, non-linear correlations, target signal (multi-metric), temporal leakage, redundancy-clustering shortlist.

Audience: a data scientist running an initial pass before building a model. The differentiator vs `ydata-profiling` / `SweetViz` / `DataPrep` is **ranking and opinion** — surface the top dozen things to look at, not 200 pages.

## Layout

```
src/biopsy/
├── io.py            # DuckDB loader, --filter expression parser, --exclude
├── stats.py         # per-column SQL aggregates, histograms, temporal buckets
├── correlations.py  # Pearson, MI, PPS (cv + holdout), Spearman, AUC, perm importance
├── clustering.py    # Spearman-distance hierarchical clustering, shortlist
├── temporal.py      # temporal leakage detection, drift, monotonicity, target drift
├── findings.py      # ranked findings synthesis
├── profile.py       # the orchestrator — entry point for the library API
├── cli.py           # Typer CLI
├── demo.py          # synthetic dataset generator (used by `biopsy demo` + tests)
├── sparkline.py     # unicode sparklines for terminal
├── render/
│   ├── terminal.py  # Rich-rendered terminal report
│   └── html.py      # Plotly + Jinja-rendered HTML report
└── templates/
    └── report.html.j2
tests/
└── test_smoke.py    # smoke tests covering core paths + the bugs from the v0.1 review
```

## Stack + conventions

- **Python 3.11+**, managed with `uv`. Source under `src/`, package name `biopsy`.
- **DuckDB-first**: column stats and Pearson correlations are computed in SQL where possible (no row transfer). Use `_quote_ident(col)` (in `io.py`) and `_quote(col)` (in `stats.py`) to safely build identifiers; use `_lit(value, is_numeric)` for literals.
- **sklearn** for MI / PPS / AUC / permutation importance. **scipy** for `spearmanr` and `ks_2samp`. **plotly + jinja2** for HTML. **rich + typer** for CLI.
- Type hints everywhere. Avoid generic `except Exception:` unless you're documenting why the failure is recoverable (it shouldn't silently turn into "no signal"; that hid a real bug — see review history).
- Severity vocabulary in findings: `critical` / `warning` / `info`. Categories: `leakage`, `suspicious`, `quality`, `distribution`, `correlation`, `target`, `temporal`.

## Things to know before changing code

1. **Filter parser** (`io.parse_filter_expr`) — symbolic ops (`==`, `!=`, `>=`, `<=`, `>`, `<`) are tried *leftmost-first* before keyword ops (`in`, `not in`, `is null`, `is not null`). Keyword ops require strict whitespace boundaries on both sides. Quoted segments are skipped. Don't reorder without rerunning `test_h3_filter_parser_symbolic_before_keyword`.

2. **`USING SAMPLE` ordering** (`io.load`) — DuckDB applies `USING SAMPLE` to the table source *before* `WHERE`. To sample *after* filtering you must wrap the filtered SELECT in a subquery. The current code does this; don't unwrap it. See `test_h1_sample_after_filter`.

3. **`_spearman`** (`correlations.py`) — uses `scipy.stats.spearmanr` with `nan_policy="omit"`. Do **not** revert to `argsort(argsort(x))` — that doesn't tie-correct and produces spurious correlations on near-constant inputs.

4. **Smart-sort for degenerate PPS** — when max PPS < 0.05 across all features, `target_signal()` falls back to sorting by `TargetSignal.best_score`. That property includes PPS / MI / AUC / perm_importance but **excludes** Spearman (matches `clustering._score_for_ranking`). If you change one, change both.

5. **Permutation-importance index alignment** (`correlations._attach_permutation_importance`) — `feat_in_target = fmask[target_mask]` is well-defined because `fmask ⊆ target_mask`. `kept_signal_idx` maps cols back to `signals` for features that had ≥30 valid rows; don't index `signals[:len(normalized)]` (that was a bug; fixed).

6. **Temporal column resolution** (`temporal.resolve_time_column`) — auto-detects when exactly one column has `kind == "temporal"`. With multiple temporal columns, emits an info finding asking for `--time`. Skipped entirely when `n_unique < 10` (quarterly snapshots don't have enough resolution).

7. **Target drift kinds** (`temporal._target_drift`) — four explicit kinds: `binary` (rate range), `multiclass` (max per-class rate range), `regression_ratio` (max/min, positive targets only), `regression_diff` (max-min, when target spans ≤ 0). Renderers in `findings.py` must handle all four.

## Run / test

```bash
# install dev environment (uv, Python 3.11+ venv)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# run the test suite (~17s)
pytest tests/

# CLI smoke
biopsy demo --rows 5000                # synthetic dataset
biopsy profile data.parquet --target y # real dataset
biopsy profile --help                  # all flags
```

## Useful flags (CLI)

| flag | purpose |
|---|---|
| `--target COL` / `-t` | target column for predictive metrics |
| `--time COL` | time column for temporal leakage (auto-detected if omitted) |
| `--exclude COL` / `-x` | drop columns from analysis (repeatable) |
| `--filter EXPR` / `-w` | filter rows (repeatable). Forms: `col in a,b,c`, `col == val`, `col >= n`, `col is not null` |
| `--sample N` | reservoir sample N rows before profiling |
| `--shortlist N` | cap the cluster-shortlist at N entries |
| `--cluster-cutoff X` | hierarchical cluster cutoff on `1−|ρ|` (default 0.30 ⇔ \|ρ\|≥0.70 collapses) |
| `--html PATH` | write HTML report |
| `--open` | open HTML in browser |

## What's NOT done (deliberately deferred)

- **`biopsy compare A.parquet B.parquet`** — explicit train-vs-eval drift mode (the right home for arbitrary holdout drift; out of scope for v0.1)
- **Notebook integration** — `Profile._repr_html_()` for inline Jupyter rendering
- **Save/load profiles** — would help on the slow real-data runs (200s on 225k × 95)
- **Regression-target end-to-end test** — `_pps_regression` is reachable but not covered by a real-data test
- **Wide-dataset assertion test** — 95-col wide dataset works in practice; no test enforces it

If you pick up any of these, follow the existing module boundaries and add a test in `tests/test_smoke.py`.

## Engram / memory

When discoveries here are worth surviving a tool switch, write to **Engram** with `project=biopsy`. Tool-native memory (Claude/Codex) is for repo-local context.
