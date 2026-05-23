"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

from biopsy.demo import write_demo_csv
from biopsy.profile import profile


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


