"""Tests — see module name for scope."""

from __future__ import annotations

import json
from pathlib import Path

from biopsy.profile import profile
from biopsy.stats import ColumnStats


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


def test_compare_report_saves_json(tmp_path: Path) -> None:
    from conftest import write_two_csvs_with_shift

    from biopsy import compare_profiles

    a_path, b_path = write_two_csvs_with_shift(tmp_path)
    report = compare_profiles(profile(a_path, target="target"), profile(b_path, target="target"))

    saved = report.save(tmp_path / "compare.json")
    payload = json.loads(saved.read_text())

    assert payload["a_name"] == "a.csv"
    assert payload["b_name"] == "b.csv"
    assert payload["schema"]["shared"]
    assert payload["drifts"]
    assert payload["findings"]


def test_categorical_drift_skips_one_sided_low_top_coverage() -> None:
    from biopsy.compare import _categorical_drift

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
