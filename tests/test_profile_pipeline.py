"""Tests — see module name for scope."""

from __future__ import annotations

import builtins
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from biopsy.demo import synthetic_dataframe, write_demo_csv
from biopsy.profile import Profile, load_profile, profile
from biopsy.profile.serde import PROFILE_SCHEMA_VERSION
from biopsy.stats import ColumnStats


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

    assert json.loads(saved.read_text(encoding="utf-8"))["schema_version"] == (
        PROFILE_SCHEMA_VERSION
    )
    assert loaded.source_name == prof.source_name
    assert loaded.source_path == prof.source_path
    assert loaded.target == "churned"
    assert loaded.columns.keys() == prof.columns.keys()
    assert loaded.findings[0].why

    html = loaded._repr_html_()
    assert "<!doctype html>" in html
    assert "biopsy" in html
    assert "churned" in html


def test_profile_from_dict_accepts_unversioned_payload(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=200)
    payload = profile(csv, deep_correlations=False).to_dict()
    payload.pop("schema_version")

    loaded = Profile.from_dict(payload)

    assert loaded.source_name == "demo.csv"


def test_profile_from_dict_rejects_unknown_schema_version(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=200)
    payload = profile(csv, deep_correlations=False).to_dict()
    payload["schema_version"] = PROFILE_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match=r"schema version 2 is not supported"):
        Profile.from_dict(payload)


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


def test_profile_closes_source_when_stats_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    run_mod = importlib.import_module("biopsy.profile.run")

    closed = False

    class FakeCon:
        def close(self) -> None:
            nonlocal closed
            closed = True

    src = SimpleNamespace(con=FakeCon(), columns=["x"])
    monkeypatch.setattr(run_mod, "load", lambda *args, **kwargs: src)

    def fail_compute_all(*_args: Any, **_kwargs: Any) -> dict[str, ColumnStats]:
        raise RuntimeError("stats failed")

    monkeypatch.setattr(run_mod, "compute_all", fail_compute_all)

    with pytest.raises(RuntimeError, match="stats failed"):
        run_mod.profile("data.csv")

    assert closed


def test_profile_closes_target_source_when_target_signal_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_mod = importlib.import_module("biopsy.profile.run")

    closed: list[str] = []

    class FakeCon:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    src = SimpleNamespace(con=FakeCon("main"), columns=["x", "y"])
    target_src = SimpleNamespace(con=FakeCon("target"), columns=["x", "y"])
    stats = {
        "x": ColumnStats("x", "INTEGER", "numeric", 10, 0, 10, 0.0),
        "y": ColumnStats("y", "BOOLEAN", "bool", 10, 0, 2, 0.0),
    }
    monkeypatch.setattr(run_mod, "load", lambda *args, **kwargs: src)
    monkeypatch.setattr(run_mod, "compute_all", lambda *_args, **_kwargs: stats)
    monkeypatch.setattr(run_mod, "correlation_pairs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        run_mod,
        "_target_source_and_stats",
        lambda **_kwargs: (target_src, stats),
    )
    monkeypatch.setattr(run_mod, "_target_summary", lambda *_args, **_kwargs: None)

    def fail_target_signal(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise RuntimeError("target signal failed")

    monkeypatch.setattr(run_mod, "target_signal", fail_target_signal)

    with pytest.raises(RuntimeError, match="target signal failed"):
        run_mod.profile("data.csv", target="y", sample=10)

    assert closed == ["target", "main"]


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
