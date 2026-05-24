"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from biopsy.demo import write_demo_csv
from biopsy.profile import profile


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
