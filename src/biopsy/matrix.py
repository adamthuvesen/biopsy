"""Shared sampled matrices for one profile run."""

from __future__ import annotations

import numpy as np

from biopsy.io import Source
from biopsy.stats import _quote


class SampleCache:
    """Cache one DuckDB reservoir sample and serve column subsets from it."""

    def __init__(self, src: Source) -> None:
        self.src = src
        self._columns: list[str] = []
        self._raw: np.ndarray | None = None
        self._max_rows = 0

    def fetch(self, columns: list[str], max_rows: int) -> tuple[list[str], np.ndarray]:
        if not columns:
            return columns, np.empty((0, 0), dtype=object)

        if (
            self._raw is not None
            and self._max_rows >= max_rows
            and set(columns).issubset(self._columns)
        ):
            idx = [self._columns.index(c) for c in columns]
            return columns, self._raw[:, idx]

        cols = columns
        if self._raw is not None:
            cols = list(dict.fromkeys([*self._columns, *columns]))

        quoted = ", ".join(_quote(c) for c in cols)
        rows = self.src.con.execute(
            f"SELECT {quoted} FROM data USING SAMPLE {max_rows} ROWS (reservoir, 42)"
        ).fetchall()
        raw = (
            np.array(rows, dtype=object)
            if rows
            else np.empty((0, len(cols)), dtype=object)
        )

        self._columns = cols
        self._raw = raw
        self._max_rows = max_rows

        idx = [self._columns.index(c) for c in columns]
        return columns, raw[:, idx]
