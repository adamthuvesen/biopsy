"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from biopsy.profile import profile


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


def test_m7_looks_like_id_no_false_positives() -> None:
    """M7: short words ending in 'id' (paid, liquid, valid) must not be flagged."""
    from biopsy.inference import looks_like_id
    assert looks_like_id("id")
    assert looks_like_id("user_id")
    assert looks_like_id("uuid")
    assert looks_like_id("session_uuid")
    # False-positive guards
    assert not looks_like_id("paid")
    assert not looks_like_id("liquid")
    assert not looks_like_id("valid")
    assert not looks_like_id("grid")
    assert not looks_like_id("revenue")


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


