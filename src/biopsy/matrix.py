"""Shared sampled matrices for one profile run."""

from __future__ import annotations

from typing import Any

import numpy as np

from biopsy.io import Source
from biopsy.stats import _quote


def _fetch_object_array(
    con: Any,
    sql: str,
    columns: list[str],
) -> np.ndarray:
    """Run `sql` and return rows as a 2D object array (cols match `columns`).

    Prefers DuckDB's `fetchnumpy()` to skip the Python-tuple intermediate;
    falls back to `fetchall()` for older bindings or non-DuckDB connections.
    """
    fetch_numpy = getattr(con.execute(sql), "fetchnumpy", None)
    if callable(fetch_numpy):
        try:
            cols_dict = fetch_numpy()
        # DuckDB binding/version differences: fall back to fetchall().
        except (TypeError, ValueError, KeyError, AttributeError):
            cols_dict = None
        if cols_dict:
            try:
                arrays = [np.asarray(cols_dict[c], dtype=object) for c in columns]
            except KeyError:
                arrays = None
            if arrays is not None:
                if arrays[0].size == 0:
                    return np.empty((0, len(columns)), dtype=object)
                return np.column_stack(arrays)
    rows = con.execute(sql).fetchall()
    if not rows:
        return np.empty((0, len(columns)), dtype=object)
    return np.array(rows, dtype=object)


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
            return columns, self._raw[:max_rows, idx]

        # Superset miss at the same max_rows: pull only the missing columns
        # (USING SAMPLE seed=42 is deterministic, so row alignment by position
        # is safe) and hstack onto the cached array instead of re-sampling.
        if self._raw is not None and self._max_rows == max_rows and self._raw.shape[0] > 0:
            missing = [c for c in columns if c not in self._columns]
            if missing:
                quoted_missing = ", ".join(_quote(c) for c in missing)
                sample_sql = (
                    f"SELECT {quoted_missing} FROM data "
                    f"USING SAMPLE {max_rows} ROWS (reservoir, 42)"
                )
                new_block = _fetch_object_array(
                    self.src.con,
                    sample_sql,
                    missing,
                )
                if new_block.shape[0] == self._raw.shape[0]:
                    self._raw = np.hstack([self._raw, new_block])
                    self._columns = [*self._columns, *missing]
                    idx = [self._columns.index(c) for c in columns]
                    return columns, self._raw[:max_rows, idx]

        cols = columns
        if self._raw is not None:
            cols = list(dict.fromkeys([*self._columns, *columns]))

        quoted = ", ".join(_quote(c) for c in cols)
        raw = _fetch_object_array(
            self.src.con,
            f"SELECT {quoted} FROM data USING SAMPLE {max_rows} ROWS (reservoir, 42)",
            cols,
        )

        self._columns = cols
        self._raw = raw
        self._max_rows = max_rows

        idx = [self._columns.index(c) for c in columns]
        return columns, raw[:max_rows, idx]
