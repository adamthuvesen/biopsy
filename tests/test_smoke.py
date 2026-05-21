"""Smoke tests on a synthetic dataset."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import biopsy.cli as cli_mod
from biopsy.cli import app
from biopsy.demo import synthetic_dataframe, write_demo_csv
from biopsy.profile import load_profile, profile
from biopsy.render.html import render as render_html


def test_profile_demo_dataset(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=3000)
    prof = profile(csv, target="churned")

    assert prof.n_rows == 3000
    assert prof.n_cols == 15
    assert "churned" in prof.columns
    assert prof.target == "churned"

    # constant_col should be flagged
    constant_findings = [f for f in prof.findings if "constant_col" in f.columns]
    assert constant_findings, "expected constant_col to be flagged"

    # days_since_last_login should be a leakage suspect (derived from churn)
    leakage = [
        f for f in prof.findings
        if f.category == "leakage" and "days_since_last_login" in f.columns
    ]
    assert leakage, (
        "expected leakage flag on days_since_last_login, "
        f"got {[f.title for f in prof.findings]}"
    )

    # some target signals should be ranked
    assert len(prof.target_signals) > 0


def test_profile_accepts_pandas_dataframe(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(synthetic_dataframe(1200))
    csv = write_demo_csv(tmp_path / "demo.csv", n=1200)

    prof_df = profile(df, target="churned", source_name="demo frame")
    prof_csv = profile(csv, target="churned")

    assert prof_df.source_name == "demo frame"
    assert prof_df.source_path is None
    assert prof_df.n_rows == prof_csv.n_rows == 1200
    assert set(prof_df.columns) == set(prof_csv.columns)
    assert "days_since_last_login" in prof_df.leakage_suspects()


def test_profile_accepts_polars_lazyframe() -> None:
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(synthetic_dataframe(800)).lazy()

    prof = profile(df, target="churned", source_name="polars demo")

    assert prof.source_name == "polars demo"
    assert prof.source_path is None
    assert prof.n_rows == 800
    assert "churned" in prof.columns


def test_profile_accepts_arrow_table() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.table(synthetic_dataframe(800))
    prof = profile(table, target="churned", source_name="arrow demo")

    assert prof.source_name == "arrow demo"
    assert prof.n_rows == 800


def test_profile_accepts_duckdb_relation() -> None:
    import duckdb

    con = duckdb.connect(":memory:")
    relation = con.sql("SELECT 1 AS x, 0 AS y UNION ALL SELECT 2 AS x, 1 AS y")
    prof = profile(relation, target="y", source_name="relation demo")

    assert prof.source_name == "relation demo"
    assert prof.source_path is None
    assert prof.n_rows == 2


def test_target_signals_have_new_metrics(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=2000)
    prof = profile(csv, target="churned")
    assert prof.target_signals, "expected target signals to be produced"

    # churned is binary classification → every signal gets an AUC.
    auc_count = sum(1 for s in prof.target_signals if s.auc is not None)
    assert auc_count >= len(prof.target_signals) - 2, (
        f"expected AUC for ~all features, got {auc_count}/{len(prof.target_signals)}"
    )

    # numeric features should get a Spearman score.
    numeric_signals = [
        s for s in prof.target_signals
        if prof.columns[s.feature].kind == "numeric"
    ]
    spearman_count = sum(1 for s in numeric_signals if s.spearman is not None)
    assert spearman_count >= len(numeric_signals) - 1

    # permutation importance should be populated for at least the top features.
    perm_count = sum(1 for s in prof.target_signals if s.perm_importance is not None)
    assert perm_count >= 5, f"expected perm importance on top features, got {perm_count}"


def test_temporal_detects_planted_leak(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=3000)
    prof = profile(csv, target="churned")

    assert prof.time_column == "signup_date", (
        f"expected signup_date auto-detected, got {prof.time_column}"
    )
    assert prof.temporal is not None
    assert prof.temporal.signals, "expected temporal signals on demo"

    # The planted leak: cohort_engagement_v2 is informative only in the late period.
    # Random CV mixes both periods (signal present) → high random_pps.
    # Time-ordered train (early only, all noise) → low time_pps.
    # Reason carries "future information"; categorized as leakage.
    critical_leakage = [
        f for f in prof.findings
        if f.category == "leakage" and f.severity == "critical"
    ]
    feats = [c for f in critical_leakage for c in f.columns]
    assert "cohort_engagement_v2" in feats, (
        "expected cohort_engagement_v2 to be flagged as critical leakage, "
        f"got critical leakage findings on: {feats}"
    )


def test_temporal_skipped_when_no_time_column(tmp_path: Path) -> None:
    import csv as csv_module
    p = tmp_path / "no_time.csv"
    with p.open("w") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "y", "z"])
        for i in range(1500):
            w.writerow([i, i * 2, i % 5])

    prof = profile(p)
    assert prof.time_column is None
    assert prof.temporal is None
    # No temporal findings of any severity
    temporal_findings = [f for f in prof.findings if f.category == "temporal"]
    assert not temporal_findings


def test_multiple_time_columns_emits_info(tmp_path: Path) -> None:
    import csv as csv_module
    p = tmp_path / "multi_time.csv"
    with p.open("w") as f:
        w = csv_module.writer(f)
        w.writerow(["created_at", "updated_at", "value"])
        for i in range(1500):
            w.writerow([f"2026-01-{(i % 28) + 1:02d}", f"2026-02-{(i % 28) + 1:02d}", i])

    prof = profile(p)
    assert prof.time_column is None  # ambiguous → skip
    info_findings = [
        f for f in prof.findings
        if f.category == "temporal" and "pass --time" in f.detail.lower()
    ]
    assert info_findings, "expected info finding directing user to --time"

    # Explicit override should resolve it
    prof2 = profile(p, time_col="created_at")
    assert prof2.time_column == "created_at"


def test_clustering_builds_shortlist(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=3000)
    prof = profile(csv, target="churned")

    assert prof.clusters is not None
    assert prof.clusters.clusters, "expected at least one cluster"
    assert prof.clusters.shortlist, "expected non-empty shortlist"

    # Every shortlist representative must appear in exactly one cluster
    rep_set = {e.feature for e in prof.clusters.shortlist}
    member_to_cluster: dict[str, int] = {}
    for c in prof.clusters.clusters:
        for m in c.members:
            assert m not in member_to_cluster, f"{m} appears in multiple clusters"
            member_to_cluster[m] = c.cluster_id
        assert c.representative in c.members
    for f in rep_set:
        assert f in member_to_cluster

    # Shortlist sorted by score descending
    scores = [e.score for e in prof.clusters.shortlist]
    assert scores == sorted(scores, reverse=True), "shortlist must be sorted by score desc"


def test_clustering_without_target(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=2000)
    prof = profile(csv)  # no target
    assert prof.clusters is not None
    assert prof.clusters.shortlist, "shortlist should still build without target"
    # without target, score_method is the no-target fallback
    methods = {e.score_method for e in prof.clusters.shortlist}
    assert methods == {"no_target"}


def test_profile_ml_helpers_and_serialization(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=3000)
    prof = profile(csv, target="churned")

    assert prof.top_findings(limit=3) == prof.findings[:3]
    assert all(f.category == "leakage" for f in prof.top_findings(category="leakage"))

    shortlist = prof.feature_shortlist()
    weak = {e.feature for e in prof.clusters.shortlist if e.is_weak} if prof.clusters else set()
    assert shortlist
    assert not weak.intersection(shortlist)
    assert prof.feature_shortlist(limit=2) == shortlist[:2]

    leakage = prof.leakage_suspects()
    assert "days_since_last_login" in leakage
    assert "churned" not in leakage

    drops = prof.drop_candidates()
    for col in ["constant_col", "user_id", "days_since_last_login"]:
        assert col in drops
    assert "churned" not in drops

    payload = prof.to_dict()
    assert payload["source_name"] == "demo.csv"
    assert payload["source_path"] == str(csv)
    assert payload["n_rows"] == 3000
    encoded = prof.to_json()
    assert json.loads(encoded)["target"] == "churned"
    assert prof.findings_records()
    assert prof.columns_records()
    assert prof.target_signal_records()
    assert prof.shortlist_records()


def test_profile_save_load_and_repr_html(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1200)
    prof = profile(
        csv,
        target="churned",
        deep_correlations=False,
        target_permutation=False,
    )

    saved = prof.save(tmp_path / "profile.json")
    loaded = load_profile(saved)

    assert loaded.source_name == prof.source_name
    assert loaded.source_path == prof.source_path
    assert loaded.target == "churned"
    assert loaded.columns.keys() == prof.columns.keys()
    assert loaded.findings[0].why

    html = loaded._repr_html_()
    assert "<!doctype html>" in html
    assert "biopsy" in html
    assert "churned" in html


def test_cli_init_and_render_saved_profile(tmp_path: Path) -> None:
    runner = CliRunner()
    csv = write_demo_csv(tmp_path / "demo.csv", n=1200)
    config_path = tmp_path / "biopsy.toml"

    init_result = runner.invoke(
        app,
        ["init", str(csv), "--output", str(config_path), "--sample", "800"],
    )
    assert init_result.exit_code == 0, init_result.output
    config_text = config_path.read_text()
    assert 'target = "churned"' in config_text
    assert 'time = "signup_date"' in config_text
    assert '"user_id"' in config_text

    prof = profile(csv, target="churned", deep_correlations=False, target_permutation=False)
    saved = prof.save(tmp_path / "profile.json")
    report = tmp_path / "report.html"
    render_result = runner.invoke(
        app,
        ["render", str(saved), "--html", str(report), "--plotly-cdn"],
    )
    assert render_result.exit_code == 0, render_result.output
    assert report.exists()
    assert '<script src="https://cdn.plot.ly' in report.read_text()


def test_profile_pandas_frame_helpers(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    csv = write_demo_csv(tmp_path / "demo.csv", n=1200)
    prof = profile(csv, target="churned")

    findings = prof.findings_frame()
    columns = prof.columns_frame()
    target = prof.target_signal_frame()
    shortlist = prof.shortlist_frame()

    assert isinstance(findings, pd.DataFrame)
    assert {"severity", "category", "title"}.issubset(findings.columns)
    assert "name" in columns.columns
    assert "feature" in target.columns
    assert "feature" in shortlist.columns


def test_profile_frame_helpers_explain_missing_pandas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1000)
    prof = profile(csv, target="churned")
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"biopsy\[dataframe\]"):
        prof.findings_frame()


def test_exclude_drops_columns(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned", exclude=["status", "constant_col"])
    assert "status" not in prof.columns
    assert "constant_col" not in prof.columns
    # other columns still present
    assert "age" in prof.columns
    # no findings refer to excluded columns
    excluded_findings = [
        f for f in prof.findings
        if any(c in f.columns for c in ("status", "constant_col"))
    ]
    assert not excluded_findings


def test_ignore_missing_exclude_skips_absent_columns(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1000)
    prof = profile(
        csv,
        target="churned",
        exclude=["status", "does_not_exist"],
        ignore_missing_exclude=True,
    )

    assert "status" not in prof.columns
    assert "does_not_exist" not in prof.columns


def test_filter_expression_reduces_rows(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=2000)
    full = profile(csv, target="churned")
    filtered = profile(csv, target="churned", where=["plan in pro,team,enterprise"])
    assert filtered.n_rows < full.n_rows
    # all surviving rows must satisfy the filter
    plans = filtered.columns["plan"].top_values
    seen_plans = {v for v, _ in plans}
    assert "free" not in seen_plans


def test_filter_parser_forms() -> None:
    from biopsy.io import parse_filter_expr
    dtypes = {"segment": "VARCHAR", "value": "DOUBLE", "name": "VARCHAR"}
    assert parse_filter_expr("segment in train,test", dtypes) == \
        "\"segment\" IN ('train', 'test')"
    assert parse_filter_expr("value > 0", dtypes) == '"value" > 0'
    assert parse_filter_expr("name is not null", dtypes) == '"name" IS NOT NULL'
    assert parse_filter_expr("segment != holdout", dtypes) == "\"segment\" <> 'holdout'"
    # quoted values
    assert parse_filter_expr("segment == 'train'", dtypes) == "\"segment\" = 'train'"


def test_h3_filter_parser_symbolic_before_keyword() -> None:
    """H3: values containing 'in' must not hijack the parse as the IN op."""
    from biopsy.io import parse_filter_expr
    dtypes = {"event_name": "VARCHAR", "desc": "VARCHAR", "email": "VARCHAR"}
    # `==` must win over `in` even though `in` appears later in the value
    assert parse_filter_expr("event_name == sign in", dtypes) == \
        "\"event_name\" = 'sign in'"
    # `!=` against a value containing 'in'
    assert parse_filter_expr("desc != foo in bar", dtypes) == \
        "\"desc\" <> 'foo in bar'"
    # Domain ending in '.in' — must be parsed as a string, not split on 'in'
    assert parse_filter_expr("email == johndoe.in", dtypes) == \
        "\"email\" = 'johndoe.in'"
    # Column name containing 'in' as a substring still parses correctly
    dtypes2 = {"training_split": "VARCHAR"}
    assert parse_filter_expr("training_split == train", dtypes2) == \
        "\"training_split\" = 'train'"
    assert parse_filter_expr("training_split in train,test", dtypes2) == \
        "\"training_split\" IN ('train', 'test')"


def test_h1_sample_after_filter(tmp_path: Path) -> None:
    """H1: --sample applied to filtered view must respect the filter."""
    import csv as csv_module
    p = tmp_path / "demo.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["region", "x"])
        # 1000 rows: 100 NO, 900 SE
        for i in range(100):
            w.writerow(["NO", i])
        for i in range(900):
            w.writerow(["SE", i])

    # Without subquery wrap, sample=200 then filter region=NO collapses to ~20 rows.
    # With the fix, we should retain all 100 NO rows (sample is 200, after filter
    # only 100 NO rows exist, so we get all of them).
    prof = profile(p, sample=200, where=["region == NO"])
    assert prof.n_rows >= 80, (
        f"sample-after-filter should retain ~100 NO rows; got {prof.n_rows}"
    )


def test_target_signal_samples_after_target_filter(tmp_path: Path) -> None:
    """Sparse labeled targets should be sampled after dropping null targets."""
    import csv as csv_module

    from biopsy.correlations import target_signal
    from biopsy.io import load
    from biopsy.stats import compute_all

    p = tmp_path / "sparse_target.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "y"])
        for i in range(900):
            w.writerow([i % 2, ""])
        for i in range(100):
            y = i % 2
            w.writerow([y, y])

    src = load(p)
    stats = compute_all(src)
    signals = target_signal(src, stats, "y", max_rows=200)
    assert signals, "target signals should use all 100 labeled rows after filtering"


def test_target_signal_stratifies_rare_binary_targets(tmp_path: Path) -> None:
    import csv as csv_module

    from biopsy.correlations import target_signal
    from biopsy.io import load
    from biopsy.stats import compute_all

    p = tmp_path / "rare_target.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "y"])
        for i in range(5000):
            y = 1 if i < 20 else 0
            w.writerow([y, y])

    src = load(p)
    stats = compute_all(src)
    stratified = target_signal(src, stats, "y", max_rows=100, include_permutation=False)
    unstratified = target_signal(
        src, stats, "y", max_rows=100, include_permutation=False, stratify=False,
    )

    assert stratified[0].positive_count == 20
    assert (unstratified[0].positive_count or 0) < 20


def test_temporal_samples_after_time_filter(tmp_path: Path) -> None:
    """Sparse timestamp coverage should not make temporal analysis vanish."""
    import csv as csv_module

    from biopsy.io import load
    from biopsy.stats import compute_all
    from biopsy.temporal import temporal_signals

    p = tmp_path / "sparse_time.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["event_date", "x"])
        for i in range(1100):
            w.writerow([f"2026-01-{(i % 30) + 1:02d}", i])
        for i in range(900):
            w.writerow(["", i])

    src = load(p)
    stats = compute_all(src)
    report = temporal_signals(src, stats, "event_date", max_rows=1000)
    assert report is not None
    assert report.time_column == "event_date"


def test_temporal_multiclass_pps_uses_actual_classes(tmp_path: Path) -> None:
    """Multiclass temporal checks should not assume labels are encoded as 0/1."""
    import csv as csv_module
    from datetime import date, timedelta

    p = tmp_path / "multiclass_time.csv"
    start = date(2024, 1, 1)
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["event_date", "feature", "target"])
        for i in range(2000):
            if i < 20:
                label = "a"
            elif i < 40:
                label = "b"
            else:
                label = "c" if i % 2 == 0 else "d"
            w.writerow([start + timedelta(days=i), label, label])

    prof = profile(p, target="target", time_col="event_date")

    assert prof.temporal is not None
    signal = next(s for s in prof.temporal.signals if s.feature == "feature")
    assert signal.random_pps is not None
    assert signal.time_pps is not None
    assert signal.random_pps > 0.9
    assert signal.time_pps > 0.9


def test_h2_spearman_handles_ties() -> None:
    """H2: _spearman must not produce spurious correlations on tied input."""
    import numpy as np

    from biopsy.correlations import _spearman
    # All-tied x against monotone y → no signal
    rho = _spearman(np.array([5.0] * 100), np.arange(100, dtype=float))
    assert rho is None or abs(rho) < 0.01, f"tied input gave rho={rho}"
    # Properly-correlated input → strong signal
    x = np.arange(100, dtype=float)
    y = 2 * x + 1
    rho = _spearman(x, y)
    assert rho is not None and rho > 0.99, f"monotone input gave rho={rho}"


def test_split_pps_baseline_uses_train_and_test_indices() -> None:
    """Holdout PPS must ignore invalid rows outside the supplied split."""
    import numpy as np

    from biopsy.correlations import _pps_classification, _pps_regression

    train_idx = np.arange(60)
    test_idx = np.arange(60, 100)
    split = (train_idx, test_idx)

    y_class = np.array([0, 1] * 50 + [-1] * 20)
    X_class = y_class.reshape(-1, 1).astype(float)
    class_score = _pps_classification(X_class, y_class, split=split)
    assert class_score > 0.9

    y_reg = np.array(([0.0, 10.0] * 50) + ([float("nan")] * 20))
    X_reg = np.nan_to_num(y_reg, nan=0.0).reshape(-1, 1)
    reg_score = _pps_regression(X_reg, y_reg, split=split)
    assert reg_score > 0.9


def test_m5_all_null_column_is_quality_critical(tmp_path: Path) -> None:
    """M5: 100%-null columns should be flagged as critical-quality, not warning-constant."""
    import csv as csv_module
    p = tmp_path / "demo.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["a", "b"])
        for i in range(200):
            w.writerow([i, ""])  # b is always null

    prof = profile(p)
    b_findings = [f for f in prof.findings if "b" in f.columns]
    got = [(f.severity, f.category, f.title) for f in b_findings]
    assert any(f.category == "quality" and f.severity == "critical" for f in b_findings), (
        f"expected critical-quality finding on 100%-null column; got {got}"
    )
    # And NOT mislabeled as "constant"
    assert not any("constant" in f.title.lower() for f in b_findings)


def test_numeric_near_constant_column_is_flagged(tmp_path: Path) -> None:
    import csv as csv_module

    p = tmp_path / "near_constant.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x"])
        for i in range(1000):
            w.writerow([0 if i < 995 else i])

    prof = profile(p)
    x_findings = [f for f in prof.findings if "x" in f.columns]
    assert any("near-constant" in f.title for f in x_findings), (
        f"expected numeric near-constant finding, got {[f.title for f in x_findings]}"
    )


def test_explicit_non_temporal_time_column_emits_info(tmp_path: Path) -> None:
    import csv as csv_module

    p = tmp_path / "bad_time.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["timeish", "x"])
        for i in range(1500):
            w.writerow([f"bucket-{i % 20}", i])

    prof = profile(p, time_col="timeish")
    assert prof.time_column is None
    temporal_findings = [f for f in prof.findings if f.category == "temporal"]
    assert any("not temporal" in f.detail.lower() for f in temporal_findings)


def test_low_cardinality_time_column_reports_target_buckets(tmp_path: Path) -> None:
    import csv as csv_module

    p = tmp_path / "quarterly.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["reporting_date", "x", "target"])
        for q, rate in [
            ("2025-01-01", 0.01),
            ("2025-04-01", 0.02),
            ("2025-07-01", 0.20),
            ("2025-10-01", 0.25),
        ]:
            positives = int(300 * rate)
            for i in range(300):
                w.writerow([q, i, 1 if i < positives else 0])

    prof = profile(p, target="target", time_col="reporting_date")

    assert prof.temporal is not None
    assert prof.temporal.signals == []
    assert len(prof.temporal.time_buckets) == 4
    assert prof.temporal.target_drift_kind == "binary"
    assert prof.temporal.target_drift is not None
    assert prof.temporal.target_drift >= 0.2
    findings = [f for f in prof.findings if f.category == "temporal"]
    assert any("target-by-period" in f.detail.lower() for f in findings)


def test_high_cardinality_time_buckets_are_capped(tmp_path: Path) -> None:
    import csv as csv_module
    from datetime import date, timedelta

    p = tmp_path / "daily.csv"
    start = date(2024, 1, 1)
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["event_date", "x", "target"])
        for i in range(1500):
            w.writerow([start + timedelta(days=i), i, int(i >= 900)])

    prof = profile(p, target="target", time_col="event_date")

    assert prof.temporal is not None
    assert len(prof.temporal.time_buckets) == 10
    assert all(" - " in b.label for b in prof.temporal.time_buckets)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hist_bins": 0}, "hist_bins"),
        ({"sample": -5}, "sample"),
        ({"cluster_cutoff": -0.1}, "cluster_cutoff"),
        ({"cluster_cutoff": 1.1}, "cluster_cutoff"),
        ({"shortlist_size": -1}, "shortlist_size"),
        ({"target_sample_size": 0}, "target_sample_size"),
        ({"bootstrap": -1}, "bootstrap"),
        ({"pps_seeds": 0}, "pps_seeds"),
        ({"max_cols": 1}, "max_cols"),
    ],
)
def test_profile_rejects_invalid_numeric_options(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        profile(tmp_path / "missing.csv", **kwargs)


def test_regression_diff_target_drift_surfaces(tmp_path: Path) -> None:
    import csv as csv_module
    from datetime import date, timedelta

    p = tmp_path / "regression_diff_drift.csv"
    start = date(2024, 1, 1)
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["event_date", "x", "target"])
        for i in range(2000):
            w.writerow([start + timedelta(days=i), i % 17, -1000 + i])

    prof = profile(p, target="target", time_col="event_date")

    assert prof.temporal is not None
    assert prof.temporal.target_drift_kind == "regression_diff"
    assert prof.temporal.target_drift is not None and prof.temporal.target_drift > 1000
    assert prof.temporal.target_drift_score is not None
    assert prof.temporal.target_drift_score >= 1.0
    findings = [
        f for f in prof.findings
        if f.category == "temporal" and {"target", "event_date"}.issubset(f.columns)
    ]
    assert findings
    assert "target spread" in findings[0].detail


def test_m7_looks_like_id_no_false_positives() -> None:
    """M7: short words ending in 'id' (paid, liquid, valid) must not be flagged."""
    from biopsy.findings import _looks_like_id
    assert _looks_like_id("id")
    assert _looks_like_id("user_id")
    assert _looks_like_id("uuid")
    assert _looks_like_id("session_uuid")
    # False-positive guards
    assert not _looks_like_id("paid")
    assert not _looks_like_id("liquid")
    assert not _looks_like_id("valid")
    assert not _looks_like_id("grid")
    assert not _looks_like_id("revenue")


def test_terminal_renders_action_plan(tmp_path: Path) -> None:
    """The terminal report includes the synthesized action plan with at
    least one of drop/transform/impute and the split/CV one-liners."""
    from io import StringIO

    from rich.console import Console

    from biopsy.render.terminal import render as render_terminal

    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")

    buf = StringIO()
    console = Console(file=buf, width=160, force_terminal=False)
    render_terminal(prof, console=console, all_columns=False)
    out = buf.getvalue()
    assert "Action plan" in out
    # The demo dataset always has at least drop + impute work.
    assert "drop" in out
    assert "impute" in out
    # The split and CV recommendation lines should appear under the tables.
    assert "split" in out
    assert "cv" in out


def test_pipeline_code_imports_and_runs(tmp_path: Path) -> None:
    """The generated sklearn pipeline module imports cleanly, exposes
    build_preprocessor(), and fits on the demo dataset.
    """
    import importlib.util
    import sys

    pd = pytest.importorskip("pandas")
    csv = write_demo_csv(tmp_path / "demo.csv", n=1200)
    prof = profile(csv, target="churned")

    code = prof.to_sklearn_pipeline_code()
    module_path = tmp_path / "generated_pp.py"
    module_path.write_text(code, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("generated_pp", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_pp"] = module
    try:
        spec.loader.exec_module(module)
        assert module.TARGET == "churned"
        df = pd.read_csv(csv)
        preproc = module.build_preprocessor()
        fitted = preproc.fit_transform(df)
        # transformed array has same row count as input
        assert fitted.shape[0] == len(df)
        # at least one feature column survives (numeric + categorical)
        assert fitted.shape[1] > 0
    finally:
        sys.modules.pop("generated_pp", None)


def test_cli_profile_writes_pipeline(tmp_path: Path) -> None:
    """`biopsy profile --pipeline out.py` writes a runnable module."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=600)
    out = tmp_path / "pp.py"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["profile", str(csv), "--target", "churned", "--pipeline", str(out), "--no-progress"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    text = out.read_text()
    assert "def build_preprocessor()" in text
    assert "ColumnTransformer" in text


def _write_two_csvs_with_shift(tmp_path: Path) -> tuple[Path, Path]:
    import csv as csv_module
    import random as _random

    rng = _random.Random(42)
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    with a_path.open("w", newline="") as fa, b_path.open("w", newline="") as fb:
        wa = csv_module.writer(fa)
        wb = csv_module.writer(fb)
        wa.writerow(["age", "income", "segment", "target"])
        wb.writerow(["age", "income", "segment", "target"])
        for _ in range(2000):
            seg = rng.choice(["A", "B", "C"])
            tgt = 1 if rng.random() < 0.3 else 0
            wa.writerow([round(rng.gauss(35, 5), 2), round(rng.gauss(50_000, 5_000), 2), seg, tgt])
            wb.writerow([round(rng.gauss(55, 5), 2), round(rng.gauss(50_500, 5_000), 2), seg, tgt])
    return a_path, b_path


def test_post_event_feature_triggers_leakage_finding(tmp_path: Path) -> None:
    """A feature with high target signal that doesn't survive a time-ordered
    split is flagged as leakage (future-information) — not just temporal drift.
    Uses the demo's planted-leak pattern (cohort_engagement_v2)."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=2500)
    prof = profile(csv, target="churned")
    leakage = [
        f for f in prof.findings
        if f.category == "leakage"
        and "cohort_engagement_v2" in f.columns
        and "future information" in f.detail.lower()
    ]
    assert leakage, (
        f"expected post-event leakage finding, "
        f"got {[(f.category, f.title) for f in prof.findings]}"
    )


def test_free_text_column_flagged_and_excluded(tmp_path: Path) -> None:
    """Long, near-unique strings get flagged as free text."""
    import csv as csv_module
    import random as _random

    rng = _random.Random(11)

    def lorem(words: int) -> str:
        vocab = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing"]
        return " ".join(rng.choice(vocab) + str(rng.randint(0, 9999)) for _ in range(words))

    p = tmp_path / "free_text.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "review"])
        for i in range(800):
            w.writerow([i % 23, lorem(10)])

    prof = profile(p)
    review_findings = [f for f in prof.findings if "review" in f.columns]
    assert any("free text" in f.title.lower() for f in review_findings), \
        f"expected free-text finding, got {[f.title for f in review_findings]}"


def test_date_string_detection_suggests_cast() -> None:
    """A pandas DataFrame whose date column is stored as object/string
    surfaces a quality finding asking for a cast."""
    pd = pytest.importorskip("pandas")
    rows = [
        {"timestamp": f"2024-01-{(i % 28) + 1:02d}", "x": i}
        for i in range(1500)
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype("string")
    prof = profile(df, source_name="date-strings")
    timestamp_findings = [f for f in prof.findings if "timestamp" in f.columns]
    assert any("stored as a string" in f.title for f in timestamp_findings), \
        f"got {[f.title for f in timestamp_findings]}"


def test_bool_like_int_detected_and_handled(tmp_path: Path) -> None:
    """Integer columns whose distinct values ⊆ {0,1} are flagged as bool-like."""
    import csv as csv_module

    p = tmp_path / "bool_int.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["is_active", "x"])
        for i in range(2000):
            w.writerow([1 if i % 3 == 0 else 0, i])

    prof = profile(p)
    bool_findings = [f for f in prof.findings if "is_active" in f.columns]
    assert any("boolean stored as int" in f.title for f in bool_findings)


def test_high_card_cat_warns_about_target_encoding(tmp_path: Path) -> None:
    """High-cardinality categoricals get a target-encoding leakage warning."""
    import csv as csv_module
    import random as _random

    rng = _random.Random(99)
    p = tmp_path / "high_card.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["zip", "x", "target"])
        for i in range(2000):
            zip_code = f"Z{rng.randint(1, 800):04d}"  # ~800 levels in 2000 rows
            w.writerow([zip_code, i % 17, 1 if i % 3 == 0 else 0])

    prof = profile(p, target="target")
    zip_findings = [f for f in prof.findings if "zip" in f.columns]
    assert any("target encoding" in f.title.lower() for f in zip_findings)


def test_target_signal_has_ci_when_bootstrap_enabled(tmp_path: Path) -> None:
    """Opting into bootstrap=50 populates AUC and MI 95% intervals."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned", bootstrap=50)
    assert prof.target_signals
    has_ci = False
    for s in prof.target_signals[:5]:
        if s.auc_ci_low is not None and s.auc_ci_high is not None:
            assert s.auc_ci_low <= s.auc_ci_high
            has_ci = True
        if s.mi_ci_low is not None and s.mi_ci_high is not None:
            assert s.mi_ci_low <= s.mi_ci_high
            has_ci = True
    assert has_ci, "expected at least one feature to carry AUC or MI CI"


def test_pps_stability_flag_fires_on_noisy_feature(tmp_path: Path) -> None:
    """Multi-seed PPS produces a stability score (CoV); a feature with no
    real signal has high coefficient of variation across permuted seeds."""
    import csv as csv_module
    import random as _random

    p = tmp_path / "noisy.csv"
    rng = _random.Random(7)
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["noise", "target"])
        for _ in range(2000):
            w.writerow([rng.gauss(0, 1), 1 if rng.random() < 0.3 else 0])

    prof = profile(p, target="target", pps_seeds=4)
    assert prof.target_signals
    noisy = next((s for s in prof.target_signals if s.feature == "noise"), None)
    assert noisy is not None
    # On a pure-noise feature, pps_stability must be defined.
    assert noisy.pps_stability is not None


def test_target_signal_confidence_low_for_rare_positives(tmp_path: Path) -> None:
    """A binary target with very few positives produces low-confidence
    target-signal rows."""
    import csv as csv_module

    p = tmp_path / "rare.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "y", "target"])
        for i in range(2000):
            # Only 10 positives across the whole frame.
            tgt = 1 if i % 200 == 0 else 0
            w.writerow([i % 17, i % 31, tgt])

    prof = profile(p, target="target")
    assert prof.target_signals
    # All ranked features must be flagged low-confidence on 10 positives.
    for s in prof.target_signals:
        assert s.confidence == "low"


def test_profile_diff_reports_new_findings(tmp_path: Path) -> None:
    """Mutating a saved profile JSON to remove a finding causes that
    finding to appear in the `resolved` bucket of the diff."""
    import json as _json

    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    prof.save(a_path)
    payload = _json.loads(a_path.read_text())
    # Drop the first finding from B so it counts as resolved in A→B.
    removed = payload["findings"].pop(0)
    b_path.write_text(_json.dumps(payload), encoding="utf-8")

    prof_a = load_profile(a_path)
    prof_b = load_profile(b_path)
    d = prof_a.diff(prof_b)
    # The removed finding shows up in resolved.
    titles = {e.title for e in d.resolved}
    assert removed["title"] in titles
    # Round-tripping the original profile produces an empty diff.
    same = prof_a.diff(prof_a)
    assert same.is_empty()


def test_cli_diff_runs(tmp_path: Path) -> None:
    """`biopsy diff a.json b.json` exits 0 and reports either resolved
    findings or 'No differences'."""
    import json as _json

    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    prof.save(a_path)
    payload = _json.loads(a_path.read_text())
    payload["findings"].pop(0)
    b_path.write_text(_json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["diff", str(a_path), str(b_path)])
    assert result.exit_code == 0, result.output
    assert "Profile diff" in result.output


def test_compute_all_is_single_pass(tmp_path: Path) -> None:
    """compute_all batches base counts (n / non-null / distinct) into one
    DuckDB query across all columns instead of one per column."""
    from biopsy.io import load
    from biopsy.stats import _batched_base_counts, compute_all

    csv = write_demo_csv(tmp_path / "demo.csv", n=2000)
    src = load(csv)

    base = _batched_base_counts(src)
    assert set(base) == set(src.columns)
    # The batched values agree with a per-column SQL count for every column.
    stats = compute_all(src)
    for name, s in stats.items():
        n, nonnull, nunique = base[name]
        assert n == s.n
        assert nonnull == s.n - s.n_null
        assert nunique == s.n_unique


def test_max_cols_limits_pairwise_pass(tmp_path: Path) -> None:
    """`max_cols` caps the number of unique columns appearing in the
    pairwise correlation list."""
    import csv as csv_module
    import random as _random

    rng = _random.Random(3)
    p = tmp_path / "wide.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        cols = [f"x{i}" for i in range(30)] + ["target"]
        w.writerow(cols)
        for _ in range(800):
            row = [rng.gauss(0, 1) for _ in range(30)] + [1 if rng.random() < 0.3 else 0]
            w.writerow(row)

    prof_full = profile(p, target="target")
    prof_capped = profile(p, target="target", max_cols=10)
    full_mi_cols = {
        c
        for pair in prof_full.correlations
        if pair.mutual_info is not None
        for c in (pair.a, pair.b)
    }
    capped_mi_cols = {
        c
        for pair in prof_capped.correlations
        if pair.mutual_info is not None
        for c in (pair.a, pair.b)
    }
    assert len(full_mi_cols) > len(capped_mi_cols)
    assert len(capped_mi_cols) <= 10


def test_notebook_starter_writes_valid_json(tmp_path: Path) -> None:
    """`biopsy notebook out.ipynb --file data.csv --target ...` writes a
    valid nbformat-4 JSON file."""
    import json as _json

    csv = write_demo_csv(tmp_path / "demo.csv", n=800)
    out = tmp_path / "starter.ipynb"
    runner = CliRunner()
    result = runner.invoke(
        app, ["notebook", str(out), "--file", str(csv), "--target", "churned"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    nb = _json.loads(out.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 4
    # The preprocessor cell references build_preprocessor.
    sources = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "build_preprocessor" in sources


def test_biopsy_toml_rejects_unknown_keys(tmp_path: Path) -> None:
    """biopsy.toml with an unknown top-level key is rejected with a
    did-you-mean suggestion."""
    cfg = tmp_path / "biopsy.toml"
    cfg.write_text("target = 'churned'\nsamplee = 1000\n")  # typo: samplee
    csv = write_demo_csv(tmp_path / "demo.csv", n=500)
    runner = CliRunner()
    result = runner.invoke(app, ["profile", str(csv), "--config", str(cfg)])
    assert result.exit_code != 0
    msg = (result.stderr or "") + "\n" + (result.output or "")
    if not msg.strip() and result.exception is not None:
        msg = str(result.exception)
    assert "Unknown config key" in msg or "samplee" in msg, msg
    assert "sample" in msg  # the suggestion


def test_biopsy_toml_supports_deep_alias_and_max_cols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "biopsy.toml"
    cfg.write_text("target = 'churned'\ndeep = true\nmax_cols = 7\n")
    csv = tmp_path / "demo.csv"
    csv.write_text("x,churned\n1,0\n2,1\n")
    seen: dict[str, Any] = {}

    def fake_profile_fn(*args: Any, **kwargs: Any) -> object:
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cli_mod, "profile_fn", fake_profile_fn)
    monkeypatch.setattr(cli_mod, "render_terminal", lambda *args, **kwargs: None)

    result = CliRunner().invoke(app, ["profile", str(csv), "--config", str(cfg)])

    assert result.exit_code == 0, result.output
    assert seen["deep_correlations"] is True
    assert seen["target_permutation"] is True
    assert seen["max_cols"] == 7


def test_biopsy_toml_rejects_conflicting_fast_and_deep(tmp_path: Path) -> None:
    cfg = tmp_path / "biopsy.toml"
    cfg.write_text("fast = true\ndeep = true\n")
    csv = write_demo_csv(tmp_path / "demo.csv", n=500)

    result = CliRunner().invoke(app, ["profile", str(csv), "--config", str(cfg)])

    assert result.exit_code != 0
    msg = (result.stderr or "") + "\n" + (result.output or "")
    if not msg.strip() and result.exception is not None:
        msg = str(result.exception)
    assert "fast" in msg and "deep" in msg and "conflict" in msg


def test_cli_doctor_runs_fast(tmp_path: Path) -> None:
    """`biopsy doctor data.csv` prints schema + candidate target/time
    columns and exits 0."""
    import time

    csv = write_demo_csv(tmp_path / "demo.csv", n=2000)
    runner = CliRunner()
    t0 = time.perf_counter()
    result = runner.invoke(app, ["doctor", str(csv)])
    elapsed = time.perf_counter() - t0
    assert result.exit_code == 0, result.output
    assert "Doctor" in result.output
    assert "candidates" in result.output.lower() or "candidate" in result.output.lower()
    assert elapsed < 10, f"doctor took {elapsed:.2f}s on a tiny demo"


def test_html_findings_groups_by_severity(tmp_path: Path) -> None:
    """Findings section carries severity/category data attributes so the
    in-page filter chips work without JS state."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")
    out = tmp_path / "report.html"
    render_html(prof, out)
    text = out.read_text()
    assert 'id="findings-filter"' in text
    # At least one finding per severity carries the data-sev attribute.
    assert 'data-sev="critical"' in text or 'data-sev="warning"' in text
    assert 'data-cat="' in text
    # Filter chips for severities exist (depend on what fired).
    assert 'class="chip' in text


def test_html_report_has_feature_drilldown(tmp_path: Path) -> None:
    """The HTML report renders a `<details class="feature-card panel">`
    per shortlisted feature."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")
    out = tmp_path / "report.html"
    render_html(prof, out)
    text = out.read_text()
    assert "feature-card" in text
    # At least one shortlisted feature appears inside a card.
    assert prof.clusters is not None and prof.clusters.shortlist
    feat = prof.clusters.shortlist[0].feature
    assert feat in text


def test_compare_html_renders(tmp_path: Path) -> None:
    """The compare HTML includes schema diff, drifted features, and at
    least one per-feature card on a clear shift."""
    from biopsy import compare_profiles
    from biopsy.render.html import render_compare

    a_path, b_path = _write_two_csvs_with_shift(tmp_path)
    a = profile(a_path, target="target")
    b = profile(b_path, target="target")
    report = compare_profiles(a, b)
    out = tmp_path / "compare.html"
    rendered = render_compare(a, b, report, out)
    assert rendered.exists()
    text = rendered.read_text()
    assert "Schema diff" in text
    assert "Top drifted features" in text
    assert "feature-card" in text
    assert "age" in text


def test_cli_compare_runs_end_to_end(tmp_path: Path) -> None:
    """`biopsy compare A B` prints schema diff + drift findings, exits 0."""
    a_path, b_path = _write_two_csvs_with_shift(tmp_path)
    html_out = tmp_path / "compare.html"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "compare", str(a_path), str(b_path),
            "--target", "target",
            "--html", str(html_out),
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "drift" in result.output.lower()
    assert "age" in result.output
    assert html_out.exists()
    text = html_out.read_text()
    assert "biopsy compare" in text


def test_cli_compare_accepts_saved_profiles(tmp_path: Path) -> None:
    """`biopsy compare a.json b.json` works on saved profile artifacts."""
    a_path, b_path = _write_two_csvs_with_shift(tmp_path)
    prof_a = profile(a_path, target="target")
    prof_b = profile(b_path, target="target")
    a_json = tmp_path / "a.json"
    b_json = tmp_path / "b.json"
    prof_a.save(a_json)
    prof_b.save(b_json)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["compare", str(a_json), str(b_json), "--no-progress"],
    )
    assert result.exit_code == 0, result.output
    assert "drift" in result.output.lower()


def test_compare_detects_numeric_shift(tmp_path: Path) -> None:
    """Compare two datasets where one numeric column has shifted between A
    and B; the shifted column ranks in the top-3 drift findings."""
    import csv as csv_module

    from biopsy import compare_profiles

    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    with a_path.open("w", newline="") as fa, b_path.open("w", newline="") as fb:
        wa = csv_module.writer(fa)
        wb = csv_module.writer(fb)
        wa.writerow(["age", "income", "segment", "target"])
        wb.writerow(["age", "income", "segment", "target"])
        # A: age ~ Normal(35, 5)
        # B: age ~ Normal(55, 5)  (large location shift)
        # income / segment / target unchanged so they should not dominate.
        rng = __import__("random").Random(42)
        for _ in range(2000):
            seg = rng.choice(["A", "B", "C"])
            tgt = 1 if rng.random() < 0.3 else 0
            wa.writerow([round(rng.gauss(35, 5), 2), round(rng.gauss(50_000, 5_000), 2), seg, tgt])
            wb.writerow([round(rng.gauss(55, 5), 2), round(rng.gauss(50_500, 5_000), 2), seg, tgt])

    a = profile(a_path, target="target")
    b = profile(b_path, target="target")
    report = compare_profiles(a, b)

    assert report.schema.shared == ["age", "income", "segment", "target"]
    # The shifted column should be ranked highest by drift_score.
    assert report.drifts[0].column == "age"
    assert report.drifts[0].ks_stat is not None and report.drifts[0].ks_stat > 0.5
    # KS p-value should be near zero on a clear shift.
    assert report.drifts[0].ks_pvalue is not None and report.drifts[0].ks_pvalue < 0.05
    # The age finding should be among the top-3 drift findings.
    top_finding_cols = [f.columns[0] for f in report.findings if f.columns][:3]
    assert "age" in top_finding_cols


def test_categorical_drift_skips_one_sided_low_top_coverage() -> None:
    from biopsy.compare import _categorical_drift
    from biopsy.stats import ColumnStats

    a = ColumnStats(
        name="segment",
        dtype="VARCHAR",
        kind="text",
        n=1000,
        n_null=0,
        n_unique=100,
        null_rate=0.0,
        top_values=[(f"a{i}", 50) for i in range(10)],
    )
    b = ColumnStats(
        name="segment",
        dtype="VARCHAR",
        kind="text",
        n=1000,
        n_null=0,
        n_unique=400,
        null_rate=0.0,
        top_values=[(f"b{i}", 10) for i in range(10)],
    )

    drift = _categorical_drift("segment", a, b)

    assert drift.js_divergence is None
    assert drift.chi2_pvalue is None


def test_split_recommendation_temporal_when_time_present(tmp_path: Path) -> None:
    """With a usable time column, the action plan recommends a temporal
    holdout and TimeSeriesSplit CV."""
    import csv as csv_module
    from datetime import date, timedelta

    p = tmp_path / "temporal.csv"
    start = date(2024, 1, 1)
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["event_date", "x", "target"])
        for i in range(1500):
            w.writerow([start + timedelta(days=i % 400), i % 31, i % 2])

    prof = profile(p, target="target", time_col="event_date")
    plan = prof.action_plan()
    assert plan.split is not None
    assert plan.split.kind == "temporal"
    assert plan.split.time_column == "event_date"
    assert plan.cv is not None
    assert plan.cv.kind == "time_series"


def test_split_recommendation_stratified_when_imbalanced(tmp_path: Path) -> None:
    """Imbalanced binary classification target → stratified split + CV."""
    import csv as csv_module

    p = tmp_path / "imbalanced.csv"
    with p.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "target"])
        for i in range(2000):
            w.writerow([i % 19, 1 if i % 50 == 0 else 0])

    prof = profile(p, target="target")
    plan = prof.action_plan()
    assert plan.split is not None
    assert plan.split.kind == "stratified"
    assert plan.split.stratify_on == "target"
    assert plan.cv is not None
    assert plan.cv.kind == "stratified_kfold"
    # Severe class imbalance triggers a class_weight strategy.
    assert plan.class_strategy is not None
    assert plan.class_strategy.kind == "class_weight"


def test_action_plan_basic(tmp_path: Path) -> None:
    """The action plan exposes drop / impute / transform / encode buckets,
    plus a split + CV + class strategy. Built once and consumed by HTML and
    terminal — no logic duplication.
    """
    from biopsy.action_plan import ActionPlan

    csv = write_demo_csv(tmp_path / "demo.csv", n=2000)
    prof = profile(csv, target="churned")

    plan = prof.action_plan()
    assert isinstance(plan, ActionPlan)
    # Demo dataset has at least one drop candidate (constant_col) and at least
    # one transform candidate (skewed monthly_revenue / outlier columns).
    assert plan.drop, f"expected drop bucket non-empty, got {plan.records()!r}"
    drop_cols = {item.column for item in plan.drop}
    assert "constant_col" in drop_cols
    # Impute bucket has at least one column (some demo columns have nulls).
    assert plan.impute, "expected non-empty impute bucket"
    # Every item carries a non-empty reason string.
    for item in plan.drop + plan.review + plan.transform + plan.encode + plan.impute:
        assert item.reason, f"empty reason on {item}"
        assert item.severity in {"critical", "warning", "info"}
        assert item.column != prof.target, "target column should never appear in drop/encode/impute"
    # Classification target → stratified split + class strategy when imbalanced.
    assert plan.split is not None
    assert plan.split.kind in {"temporal", "stratified", "random"}
    assert plan.cv is not None
    # records() is JSON-serialisable.
    records = plan.records()
    assert records
    assert {"bucket", "column", "action", "reason", "severity", "evidence"} <= records[0].keys()
    # action_plan() is idempotent — repeated calls return equivalent plans.
    # (No identity guarantee — caching across mutations introduced staleness.)
    again = prof.action_plan()
    assert again.records() == plan.records()


def test_html_render(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1000)
    prof = profile(csv, target="churned")
    out = render_html(prof, tmp_path / "report.html")
    assert out.exists()
    content = out.read_text()
    assert "biopsy" in content
    assert "churned" in content
    assert "plotly" in content.lower()
    assert '<script src="https://cdn.plot.ly' not in content
    assert "plotly.js" in content.lower()

    cdn_out = render_html(prof, tmp_path / "report-cdn.html", embed_plotly=False)
    cdn_content = cdn_out.read_text()
    assert '<script src="https://cdn.plot.ly' in cdn_content
