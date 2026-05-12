"""Smoke tests on a synthetic dataset."""

from __future__ import annotations

from pathlib import Path

from sketch.demo import write_demo_csv
from sketch.profile import profile
from sketch.render.html import render as render_html


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
    critical_temporal = [
        f for f in prof.findings
        if f.category == "temporal" and f.severity == "critical"
    ]
    feats = [c for f in critical_temporal for c in f.columns]
    assert "cohort_engagement_v2" in feats, (
        "expected cohort_engagement_v2 to be flagged as temporal-critical, "
        f"got critical temporal findings on: {feats}"
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
    from sketch.io import parse_filter_expr
    dtypes = {"REGION": "VARCHAR", "value": "DOUBLE", "name": "VARCHAR"}
    assert parse_filter_expr("segment in train,test", dtypes) == \
        "\"REGION\" IN ('train', 'test')"
    assert parse_filter_expr("value > 0", dtypes) == '"value" > 0'
    assert parse_filter_expr("name is not null", dtypes) == '"name" IS NOT NULL'
    assert parse_filter_expr("segment != holdout", dtypes) == "\"REGION\" <> 'train'"
    # quoted values
    assert parse_filter_expr("segment == 'train'", dtypes) == "\"REGION\" = 'train'"


def test_h3_filter_parser_symbolic_before_keyword() -> None:
    """H3: values containing 'in' must not hijack the parse as the IN op."""
    from sketch.io import parse_filter_expr
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


def test_h2_spearman_handles_ties() -> None:
    """H2: _spearman must not produce spurious correlations on tied input."""
    import numpy as np
    from sketch.correlations import _spearman
    # All-tied x against monotone y → no signal
    rho = _spearman(np.array([5.0] * 100), np.arange(100, dtype=float))
    assert rho is None or abs(rho) < 0.01, f"tied input gave rho={rho}"
    # Properly-correlated input → strong signal
    x = np.arange(100, dtype=float)
    y = 2 * x + 1
    rho = _spearman(x, y)
    assert rho is not None and rho > 0.99, f"monotone input gave rho={rho}"


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
    assert any(f.category == "quality" and f.severity == "critical" for f in b_findings), (
        f"expected critical-quality finding on 100%-null column; got {[(f.severity,f.category,f.title) for f in b_findings]}"
    )
    # And NOT mislabeled as "constant"
    assert not any("constant" in f.title.lower() for f in b_findings)


def test_m7_looks_like_id_no_false_positives() -> None:
    """M7: short words ending in 'id' (paid, liquid, valid) must not be flagged."""
    from sketch.findings import _looks_like_id
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


def test_html_render(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1000)
    prof = profile(csv, target="churned")
    out = render_html(prof, tmp_path / "report.html")
    assert out.exists()
    content = out.read_text()
    assert "sketch" in content
    assert "churned" in content
    assert "plotly" in content.lower()
