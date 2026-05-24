"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

from biopsy.demo import write_demo_csv
from biopsy.profile import profile


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
        f for f in prof.findings if f.category == "leakage" and f.severity == "critical"
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
        f for f in prof.findings if f.category == "temporal" and "pass --time" in f.detail.lower()
    ]
    assert info_findings, "expected info finding directing user to --time"

    # Explicit override should resolve it
    prof2 = profile(p, time_col="created_at")
    assert prof2.time_column == "created_at"


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


def test_time_to_float_does_not_require_pandas() -> None:
    import numpy as np

    from biopsy.temporal import _time_to_float

    values = np.array(["2024-01-01", np.datetime64("2024-01-02"), None, "not-a-date"], dtype=object)
    out = _time_to_float(values)

    assert np.isfinite(out[:2]).all()
    assert out[1] > out[0]
    assert np.isnan(out[2])
    assert np.isnan(out[3])


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
        f
        for f in prof.findings
        if f.category == "temporal" and {"target", "event_date"}.issubset(f.columns)
    ]
    assert findings
    assert "target spread" in findings[0].detail


def test_post_event_feature_triggers_leakage_finding(tmp_path: Path) -> None:
    """A feature with high target signal that doesn't survive a time-ordered
    split is flagged as leakage (future-information) — not just temporal drift.
    Uses the demo's planted-leak pattern (cohort_engagement_v2)."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=2500)
    prof = profile(csv, target="churned")
    leakage = [
        f
        for f in prof.findings
        if f.category == "leakage"
        and "cohort_engagement_v2" in f.columns
        and "future information" in f.detail.lower()
    ]
    assert leakage, (
        f"expected post-event leakage finding, got {[(f.category, f.title) for f in prof.findings]}"
    )


def test_temporal_signal_leakage_kind_on_planted_leak(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=2500)
    prof = profile(csv, target="churned")
    assert prof.temporal is not None
    sig = next(s for s in prof.temporal.signals if s.feature == "cohort_engagement_v2")
    assert sig.leakage_kind in {"random_cv", "post_event"}
    assert sig.severity == "critical"
