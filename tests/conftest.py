"""Shared test helpers."""

from __future__ import annotations

import csv as csv_module
import random as _random
from pathlib import Path


def write_two_csvs_with_shift(tmp_path: Path) -> tuple[Path, Path]:
    rng = _random.Random(42)
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    with a_path.open("w", newline="") as fa, b_path.open("w", newline="") as fb:
        wa = csv_module.writer(fa)
        wb = csv_module.writer(fb)
        wa.writerow(["age", "income", "segment", "target"])
        wb.writerow(["age", "income", "segment", "target"])
        for _ in range(2000):
            seg = rng.choice(["A", "B", "C"])
            tgt = 1 if rng.random() < 0.3 else 0
            wa.writerow([round(rng.gauss(35, 5), 2), round(rng.gauss(50_000, 5_000), 2), seg, tgt])
            wb.writerow([round(rng.gauss(55, 5), 2), round(rng.gauss(50_500, 5_000), 2), seg, tgt])
    return a_path, b_path
