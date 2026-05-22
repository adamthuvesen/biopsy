"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

from biopsy.demo import write_demo_csv
from biopsy.profile import profile


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


