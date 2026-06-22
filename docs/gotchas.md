# Gotchas

Non-obvious algorithm and ordering invariants. Read the matching entry before
changing any of these subsystems — each one cost a real bug or has a regression
test guarding it.

1. **Filter parser** (`io.parse_filter_expr`) — symbolic ops (`==`, `!=`, `>=`,
   `<=`, `>`, `<`) are tried *leftmost-first* before keyword ops (`in`,
   `not in`, `is null`, `is not null`). Keyword ops require strict whitespace
   boundaries on both sides. Quoted segments are skipped. Don't reorder without
   rerunning `test_h3_filter_parser_symbolic_before_keyword`.

2. **`USING SAMPLE` ordering** (`io.load`) — DuckDB applies `USING SAMPLE` to
   the table source *before* `WHERE`. To sample *after* filtering, wrap the
   filtered SELECT in a subquery. See `test_h1_sample_after_filter`.

3. **`_spearman`** (`correlations.py`) — uses `scipy.stats.spearmanr` with
   `nan_policy="omit"`. Do not revert to `argsort(argsort(x))` — that doesn't
   tie-correct and produces spurious correlations on near-constant inputs.

4. **Smart-sort for degenerate PPS** — when max PPS < 0.05 across all features,
   `target_signal()` falls back to sorting by `TargetSignal.best_score`. That
   property includes PPS / MI / AUC / perm_importance but **excludes** Spearman
   (matches `clustering._score_for_ranking`). Change both together.

5. **Permutation-importance index alignment**
   (`correlations._attach_permutation_importance`) — `feat_in_target =
   fmask[target_mask]` is well-defined because `fmask ⊆ target_mask`.
   `kept_signal_idx` maps cols back to `signals` for features that had ≥30 valid
   rows; don't index `signals[:len(normalized)]`.

6. **Temporal column resolution** (`temporal.resolve_time_column`) — auto-detects
   when exactly one column has `kind == "temporal"`. With multiple, emits an
   info finding asking for `--time`. Skipped entirely when `n_unique < 10`;
   low-cardinality bucket mode handles 3–9 buckets via the bucket aggregator.

7. **Target drift kinds** (`temporal._target_drift`) — four kinds: `binary`,
   `multiclass`, `regression_ratio`, `regression_diff`. `is_target_drifted()`
   and the renderers in `findings.py` must handle all four.

8. **Temporal leakage classification** (`temporal._classify`) — sets
   `TemporalSignal.leakage_kind` (`random_cv`, `post_event`, `drift`,
   `strong_drift`, `monotonic`, or `none`). `findings.temporal_findings` maps
   `random_cv` and `post_event` to `category="leakage"`. Saved profiles without
   `leakage_kind` are back-filled via `temporal.infer_leakage_kind_from_legacy`
   in `temporal_signal_from_payload`.

9. **Action plan single-source-of-truth** (`action_plan.py`) — HTML, terminal,
   and `to_sklearn_pipeline_code()` all consume `Profile.action_plan()`. Don't
   duplicate the drop/impute/encode/transform logic anywhere else.

10. **Compare guards on noisy categoricals** (`compare._categorical_drift`) —
    skips drift computation when `n_unique / nonnull > 0.5` (IDs, free text) or
    when top-K coverage of either side is < 25% (high-card categoricals where
    top values are sampling noise).
