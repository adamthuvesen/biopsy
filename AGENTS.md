# AGENTS.md — biopsy

Project-specific instructions for AI coding agents working on this repo. The user's global `~/dotfiles/agents/AGENTS.md` applies on top — this file overrides for biopsy-specific concerns.

## What this is

`biopsy` is an opinionated EDA library + CLI. Point it at a CSV or Parquet, get a ranked report of what matters for modeling — distributions, nulls, outliers, non-linear correlations, target signal (multi-metric), temporal leakage, redundancy clustering, drift comparison, and an executable action plan.

Audience: a data scientist running an initial pass before training a model. Differentiator vs `ydata-profiling` / `SweetViz` / `DataPrep`: **ranking + opinion + action** — the top dozen things to look at, plus a runnable sklearn preprocessor and a split/CV recommendation.

## Layout

```
src/biopsy/
├── io.py            # DuckDB loader, --filter parser, --exclude
├── stats.py         # per-column SQL aggregates, histograms, batched base counts
├── correlations.py  # Pearson, MI, PPS, Spearman, AUC, perm importance, bootstrap CIs
├── clustering.py    # Spearman-distance hierarchical clustering, shortlist
├── temporal.py      # temporal leakage, drift, monotonicity, target drift
├── findings.py      # ranked findings synthesis + smart detectors
├── action_plan.py   # drop / impute / encode / transform / split / cv + sklearn codegen
├── compare.py       # profile-to-profile drift (KS / Wasserstein / PSI / χ² / JS)
├── profile.py       # orchestrator + Profile / ProfileDiff API
├── cli.py           # Typer CLI (profile / compare / diff / doctor / notebook / render / init / demo)
├── demo.py          # synthetic dataset generator
├── sparkline.py     # unicode sparklines for terminal
├── render/
│   ├── terminal.py  # Rich terminal report
│   └── html.py      # Plotly + Jinja HTML report and compare report
└── templates/
    ├── report.html.j2
    └── compare.html.j2
tests/
└── test_smoke.py    # smoke tests covering core paths + regression tests for fixed bugs
```

## Stack + conventions

- **Python 3.11+**, managed with `uv`. Source under `src/`, package name `biopsy`.
- **DuckDB-first**: column stats and Pearson correlations are computed in SQL. Use `_quote_ident(col)` (in `io.py`) and `_quote(col)` (in `stats.py`) for identifiers; `_lit(value, is_numeric)` for literals.
- **sklearn** for MI / PPS / AUC / permutation importance / bootstrap CIs. **scipy** for `spearmanr`, `ks_2samp`, `chi2_contingency`. **plotly + jinja2** for HTML. **rich + typer** for CLI.
- Type hints everywhere. Avoid generic `except Exception:` unless the failure is genuinely recoverable and documented (silent failures hid a real bug in the v0.1 review).
- Severity vocabulary: `critical` / `warning` / `info`. Categories: `leakage`, `suspicious`, `quality`, `distribution`, `correlation`, `target`, `temporal`, `drift`.

## Things to know before changing code

1. **Filter parser** (`io.parse_filter_expr`) — symbolic ops (`==`, `!=`, `>=`, `<=`, `>`, `<`) are tried *leftmost-first* before keyword ops (`in`, `not in`, `is null`, `is not null`). Keyword ops require strict whitespace boundaries on both sides. Quoted segments are skipped. Don't reorder without rerunning `test_h3_filter_parser_symbolic_before_keyword`.

2. **`USING SAMPLE` ordering** (`io.load`) — DuckDB applies `USING SAMPLE` to the table source *before* `WHERE`. To sample *after* filtering, wrap the filtered SELECT in a subquery. See `test_h1_sample_after_filter`.

3. **`_spearman`** (`correlations.py`) — uses `scipy.stats.spearmanr` with `nan_policy="omit"`. Do not revert to `argsort(argsort(x))` — that doesn't tie-correct and produces spurious correlations on near-constant inputs.

4. **Smart-sort for degenerate PPS** — when max PPS < 0.05 across all features, `target_signal()` falls back to sorting by `TargetSignal.best_score`. That property includes PPS / MI / AUC / perm_importance but **excludes** Spearman (matches `clustering._score_for_ranking`). Change both together.

5. **Permutation-importance index alignment** (`correlations._attach_permutation_importance`) — `feat_in_target = fmask[target_mask]` is well-defined because `fmask ⊆ target_mask`. `kept_signal_idx` maps cols back to `signals` for features that had ≥30 valid rows; don't index `signals[:len(normalized)]`.

6. **Temporal column resolution** (`temporal.resolve_time_column`) — auto-detects when exactly one column has `kind == "temporal"`. With multiple, emits an info finding asking for `--time`. Skipped entirely when `n_unique < 10`; low-cardinality bucket mode handles 3–9 buckets via the bucket aggregator.

7. **Target drift kinds** (`temporal._target_drift`) — four kinds: `binary`, `multiclass`, `regression_ratio`, `regression_diff`. `is_target_drifted()` and the renderers in `findings.py` must handle all four.

8. **Temporal leakage classification** (`temporal._classify`) — sets `TemporalSignal.leakage_kind` (`random_cv`, `post_event`, `drift`, `strong_drift`, `monotonic`, or `none`). `findings.temporal_findings` maps `random_cv` and `post_event` to `category="leakage"`. Saved profiles without `leakage_kind` are back-filled via `temporal.infer_leakage_kind_from_legacy` in `temporal_signal_from_payload`.

9. **Action plan single-source-of-truth** (`action_plan.py`) — HTML, terminal, and `to_sklearn_pipeline_code()` all consume `Profile.action_plan()`. Don't duplicate the drop/impute/encode/transform logic anywhere else.

10. **Compare guards on noisy categoricals** (`compare._categorical_drift`) — skips drift computation when `n_unique / nonnull > 0.5` (IDs, free text) or when top-K coverage of either side is < 25% (high-card categoricals where top values are sampling noise).

## Run / test

```bash
# install dev environment (uv, Python 3.11+ venv)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# run the test suite
uv run python -m pytest tests/ -q

# CLI smoke
biopsy demo --rows 5000                # synthetic dataset
biopsy profile data.parquet --target y # real dataset
biopsy compare a.parquet b.parquet     # drift
biopsy diff a.json b.json              # finding-level diff
biopsy doctor data.parquet             # quick schema + candidates
biopsy profile --help                  # all flags
```

## Engram / memory

When discoveries here are worth surviving a tool switch, write to **Engram** with `project=biopsy`. Tool-native memory (Claude/Codex) is for repo-local context.
