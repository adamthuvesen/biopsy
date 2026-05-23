"""Tests — see module name for scope."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import biopsy.cli as cli_mod
from biopsy.cli import app
from biopsy.demo import write_demo_csv
from biopsy.profile import profile
from biopsy.stats import ColumnStats


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


def test_cli_init_closes_local_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class FakeCon:
        def close(self) -> None:
            nonlocal closed
            closed = True

    src = SimpleNamespace(con=FakeCon())
    stats = {
        "target": ColumnStats(
            name="target", dtype="BOOLEAN", kind="bool", n=10, n_null=0, n_unique=2, null_rate=0
        )
    }
    monkeypatch.setattr(cli_mod, "load", lambda *args, **kwargs: src)
    monkeypatch.setattr(cli_mod, "compute_all", lambda *_args, **_kwargs: stats)

    result = CliRunner().invoke(
        app,
        ["init", str(tmp_path / "input.csv"), "--output", str(tmp_path / "biopsy.toml")],
    )

    assert result.exit_code == 0, result.output
    assert closed


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
    assert "TARGET_KIND = 'classification'" in sources
    assert "GradientBoostingClassifier" in sources


def test_notebook_starter_uses_regression_baseline(tmp_path: Path) -> None:
    import csv as csv_module
    import json as _json

    csv = tmp_path / "regression.csv"
    with csv.open("w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["x", "y"])
        for i in range(300):
            w.writerow([i, i * 0.5 + 1])

    out = tmp_path / "starter.ipynb"
    result = CliRunner().invoke(
        app, ["notebook", str(out), "--file", str(csv), "--target", "y"]
    )

    assert result.exit_code == 0, result.output
    nb = _json.loads(out.read_text())
    sources = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "TARGET_KIND = 'regression'" in sources
    assert "GradientBoostingRegressor" in sources
    assert "mean_absolute_error" in sources
    assert "stratify=y" not in sources


def test_notebook_starter_without_target_skips_supervised_model(tmp_path: Path) -> None:
    import json as _json

    notebook_json = cli_mod._starter_notebook(
        data_file=tmp_path / "data.csv",
        target=None,
        target_kind=None,
        time_col=None,
        pipeline_code="def build_preprocessor():\n    raise NotImplementedError\n",
        shortlist=[],
        split_detail="Random 80/20 holdout",
        cv_detail="KFold(n_splits=5)",
        class_detail=None,
    )

    nb = _json.loads(notebook_json)
    sources = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert "TARGET_KIND = None" in sources
    assert "GradientBoosting" not in sources
    assert "fit_transform(X)" in sources


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


@pytest.mark.parametrize(
    ("config_text", "expected"),
    [
        ("sample = 'nope'\n", "sample"),
        ("fast = 'false'\n", "fast"),
        ("deep = 'true'\n", "deep"),
        ("cluster_cutoff = 2.0\n", "cluster_cutoff"),
        ("max_cols = 1\n", "max_cols"),
    ],
)
def test_biopsy_toml_rejects_invalid_value_types(
    tmp_path: Path,
    config_text: str,
    expected: str,
) -> None:
    cfg = tmp_path / "biopsy.toml"
    cfg.write_text(config_text)
    csv = write_demo_csv(tmp_path / "demo.csv", n=500)

    result = CliRunner().invoke(app, ["profile", str(csv), "--config", str(cfg)])

    assert result.exit_code != 0
    msg = (result.stderr or "") + "\n" + (result.output or "")
    if not msg.strip() and result.exception is not None:
        msg = str(result.exception)
    assert expected in msg


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


def test_doctor_load_closes_local_source(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = False

    class FakeCon:
        def close(self) -> None:
            nonlocal closed
            closed = True

    src = SimpleNamespace(con=FakeCon(), source_name="demo.csv", n_rows=3)
    monkeypatch.setattr(cli_mod, "load", lambda *args, **kwargs: src)
    monkeypatch.setattr(cli_mod, "compute_all", lambda *_args, **_kwargs: {})

    stats, source_name, n_rows, schema_only = cli_mod._doctor_load(
        "demo.csv", sample=100, credentials_env=None
    )

    assert stats == {}
    assert source_name == "demo.csv"
    assert n_rows == 3
    assert schema_only is False
    assert closed


def test_cli_compare_runs_end_to_end(tmp_path: Path) -> None:
    """`biopsy compare A B` prints schema diff + drift findings, exits 0."""
    from conftest import write_two_csvs_with_shift

    a_path, b_path = write_two_csvs_with_shift(tmp_path)
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
    from conftest import write_two_csvs_with_shift

    a_path, b_path = write_two_csvs_with_shift(tmp_path)
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


def test_cli_compare_accepts_json_data_files(tmp_path: Path) -> None:
    a_json = tmp_path / "a.json"
    b_json = tmp_path / "b.json"
    a_rows = [{"x": i, "segment": "A" if i % 2 else "B"} for i in range(200)]
    b_rows = [{"x": i + 50, "segment": "A" if i % 2 else "B"} for i in range(200)]
    a_json.write_text(json.dumps(a_rows), encoding="utf-8")
    b_json.write_text(json.dumps(b_rows), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["compare", str(a_json), str(b_json), "--no-progress"],
    )

    assert result.exit_code == 0, result.output
    assert "drift" in result.output.lower()
    assert "x" in result.output


def test_cli_compare_passes_credentials_env_to_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_profile_fn(*args: Any, **kwargs: Any) -> Any:
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(source_name=f"side-{len(calls)}")

    monkeypatch.setattr(cli_mod, "profile_fn", fake_profile_fn)
    monkeypatch.setattr(cli_mod, "_print_compare", lambda *args, **kwargs: None)

    import biopsy.compare as compare_mod

    monkeypatch.setattr(compare_mod, "compare_profiles", lambda *_args: object())

    result = CliRunner().invoke(
        app,
        [
            "compare",
            "postgres://host/db?table=public.a",
            "postgres://host/db?table=public.b",
            "--credentials-env",
            "STAGING",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["kwargs"]["credentials_env"] for call in calls] == ["STAGING", "STAGING"]


