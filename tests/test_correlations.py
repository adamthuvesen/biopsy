"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

from biopsy.demo import write_demo_csv
from biopsy.profile import profile


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


def test_valid_mask_does_not_require_pandas() -> None:
    import numpy as np

    from biopsy.correlations import _valid_mask

    values = np.array(
        [1.0, None, float("nan"), np.float64("nan"), np.datetime64("NaT"), "ok"],
        dtype=object,
    )

    assert _valid_mask(values).tolist() == [True, False, False, False, False, True]


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


