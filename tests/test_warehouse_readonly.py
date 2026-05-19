"""Read-only enforcement: lint test that fails the build if any warehouse
adapter file contains a forbidden mutation token.

The contract from the design doc: adapters issue SELECTs only. They never
INSERT, UPDATE, DELETE, MERGE, TRUNCATE, DROP, ALTER, CREATE TABLE, or
CREATE OR REPLACE TABLE against the remote warehouse. The only DDL biopsy
issues is `CREATE TABLE data AS …` against its LOCAL DuckDB connection,
which happens in `biopsy.io.load()` — not in `biopsy.warehouse`.

This test greps every line under `src/biopsy/warehouse/` for the forbidden
patterns. A small whitelist covers comments and docstrings that mention
these keywords descriptively (e.g. "never issues INSERT against...").
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WAREHOUSE_DIR = Path(__file__).resolve().parent.parent / "src" / "biopsy" / "warehouse"


# Each pattern matches a remote-side mutation. Anchored on word boundary +
# the second keyword so a docstring mentioning the bare verb doesn't fire.
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "INSERT INTO":              re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    "INSERT VALUES":            re.compile(r"\bINSERT\s+\(", re.IGNORECASE),
    "UPDATE SET":               re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
    "DELETE FROM":              re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    "MERGE INTO":               re.compile(r"\bMERGE\s+INTO\b", re.IGNORECASE),
    "TRUNCATE":                 re.compile(r"\bTRUNCATE\b", re.IGNORECASE),
    "DROP TABLE/SCHEMA/etc":    re.compile(
        r"\bDROP\s+(TABLE|SCHEMA|DATABASE|VIEW)\b", re.IGNORECASE,
    ),
    "ALTER TABLE":              re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE),
    "CREATE TABLE":             re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE),
    "CREATE OR REPLACE TABLE":  re.compile(r"\bCREATE\s+OR\s+REPLACE\s+TABLE\b", re.IGNORECASE),
}


# Lines containing this marker comment are skipped — for legitimate
# descriptions or whitelisted constructs (none expected in this slice).
_WHITELIST_MARKER = "biopsy-readonly-ok"


def _scan_file(path: Path) -> list[tuple[str, int, str]]:
    """Return (pattern_name, lineno, line) for every forbidden hit."""
    hits: list[tuple[str, int, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if _WHITELIST_MARKER in raw:
            continue
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(raw):
                hits.append((name, lineno, raw.strip()))
    return hits


def test_warehouse_dir_exists() -> None:
    """Sanity check: the lint actually runs against real files."""
    assert WAREHOUSE_DIR.is_dir(), f"warehouse dir not found: {WAREHOUSE_DIR}"
    py_files = list(WAREHOUSE_DIR.rglob("*.py"))
    assert py_files, "no Python files under warehouse/ — lint would silently pass"


@pytest.mark.parametrize(
    "py_file",
    sorted(WAREHOUSE_DIR.rglob("*.py")),
    ids=lambda p: p.relative_to(WAREHOUSE_DIR).as_posix(),
)
def test_no_remote_mutation_tokens(py_file: Path) -> None:
    """Every .py file under warehouse/ MUST be free of remote-mutation tokens."""
    hits = _scan_file(py_file)
    if hits:
        formatted = "\n".join(
            f"  {py_file.name}:{lineno}  [{pattern}]  {line}"
            for pattern, lineno, line in hits
        )
        pytest.fail(
            f"forbidden remote-mutation tokens found in {py_file.name}:\n"
            f"{formatted}\n\n"
            "Adapters under src/biopsy/warehouse/ MUST be read-only. "
            "If a token is a false positive (e.g. in a docstring), add the "
            f"comment marker '{_WHITELIST_MARKER}' on the same line."
        )


def test_lint_catches_known_bad_string(tmp_path: Path) -> None:
    """Self-test: the lint regex actually fires on a real mutation token."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        'def x():\n    con.execute("INSERT INTO foo VALUES (1)")\n',
        encoding="utf-8",
    )
    hits = _scan_file(bad)
    assert hits, "lint failed to catch obvious INSERT INTO statement"
    assert hits[0][0] == "INSERT INTO"
