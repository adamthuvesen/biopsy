"""biopsy notebook — starter ipynb scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import biopsy.cli as cli


def notebook_cmd(
    output: Path = typer.Argument(..., help="Notebook (.ipynb) path to write."),
    data_file: Path = typer.Option(..., "--file", help="Data file to profile."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column."),
    time_col: str | None = typer.Option(None, "--time", help="Time column."),
) -> None:
    """Write a starter notebook scaffolded against a profile's action plan."""
    console = Console()
    prof = cli.profile_fn(data_file, target=target, time_col=time_col)
    pipeline_code = prof.to_sklearn_pipeline_code()
    plan = prof.action_plan()
    shortlist = [e.feature for e in prof.clusters.shortlist[:20]] if prof.clusters else []
    split_detail = plan.split.detail if plan.split else "Random 80/20 holdout"
    cv_detail = plan.cv.detail if plan.cv else "KFold(n_splits=5)"
    class_detail = plan.class_strategy.detail if plan.class_strategy else None
    target_kind = prof.target_summary.kind if prof.target_summary else None

    notebook_json = starter_notebook(
        data_file=data_file,
        target=target,
        target_kind=target_kind,
        time_col=time_col,
        pipeline_code=pipeline_code,
        shortlist=shortlist,
        split_detail=split_detail,
        cv_detail=cv_detail,
        class_detail=class_detail,
    )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(notebook_json, encoding="utf-8")
    console.print(f"[dim]Notebook:[/dim] {output}")


def starter_notebook(
    *,
    data_file: Path,
    target: str | None,
    target_kind: str | None,
    time_col: str | None,
    pipeline_code: str,
    shortlist: list[str],
    split_detail: str,
    cv_detail: str,
    class_detail: str | None,
) -> str:
    import json as _json

    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    def md_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    target_repr = repr(target) if target else "None"
    target_kind_repr = repr(target_kind) if target_kind else "None"
    shortlist_repr = repr(shortlist)
    extras = []
    if class_detail:
        extras.append(f"# Class strategy: {class_detail}")
    extras_block = "\n".join(extras)

    if target_kind == "classification":
        imports = (
            "import pandas as pd\n"
            "from sklearn.metrics import accuracy_score, roc_auc_score\n"
            "from sklearn.ensemble import GradientBoostingClassifier\n"
            "from sklearn.pipeline import Pipeline\n"
        )
        split_code = (
            "from sklearn.model_selection import train_test_split\n\n"
            "X = df.drop(columns=[TARGET])\n"
            "y = df[TARGET]\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42, stratify=y\n"
            ")\n"
        )
        baseline_code = (
            "# `build_preprocessor` is defined by the cell above (biopsy codegen).\n"
            "assert 'build_preprocessor' in dir(), "
            '"Pipeline cell did not define build_preprocessor()"\n'
            "preproc = build_preprocessor()\n"
            "pipe = Pipeline([\n"
            "    ('preprocess', preproc),\n"
            "    ('model', GradientBoostingClassifier(random_state=42)),\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "pred = pipe.predict(X_test)\n"
            "print('Accuracy =', accuracy_score(y_test, pred))\n"
            "if len(set(y_test)) == 2 and hasattr(pipe, 'predict_proba'):\n"
            "    proba = pipe.predict_proba(X_test)[:, 1]\n"
            "    print('AUC =', roc_auc_score(y_test, proba))\n"
        )
    elif target_kind == "regression":
        imports = (
            "import pandas as pd\n"
            "from sklearn.metrics import mean_absolute_error\n"
            "from sklearn.ensemble import GradientBoostingRegressor\n"
            "from sklearn.pipeline import Pipeline\n"
        )
        split_code = (
            "from sklearn.model_selection import train_test_split\n\n"
            "X = df.drop(columns=[TARGET])\n"
            "y = df[TARGET]\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X, y, test_size=0.2, random_state=42\n"
            ")\n"
        )
        baseline_code = (
            "# `build_preprocessor` is defined by the cell above (biopsy codegen).\n"
            "assert 'build_preprocessor' in dir(), "
            '"Pipeline cell did not define build_preprocessor()"\n'
            "preproc = build_preprocessor()\n"
            "pipe = Pipeline([\n"
            "    ('preprocess', preproc),\n"
            "    ('model', GradientBoostingRegressor(random_state=42)),\n"
            "])\n"
            "pipe.fit(X_train, y_train)\n"
            "pred = pipe.predict(X_test)\n"
            "print('MAE =', mean_absolute_error(y_test, pred))\n"
        )
    else:
        imports = "import pandas as pd\n"
        split_code = "X = df.drop(columns=[TARGET]) if TARGET else df\nX.head()\n"
        baseline_code = (
            "# `build_preprocessor` is defined by the cell above (biopsy codegen).\n"
            "assert 'build_preprocessor' in dir(), "
            '"Pipeline cell did not define build_preprocessor()"\n'
            "preproc = build_preprocessor()\n"
            "features = preproc.fit_transform(X)\n"
            "features.shape\n"
        )

    cells = [
        md_cell(
            f"# Starter notebook — {data_file.name}\n\n"
            f"Generated by `biopsy notebook`. Target: `{target}`."
        ),
        code_cell(imports),
        md_cell("## Load"),
        code_cell(
            f"DATA = {str(data_file)!r}\n"
            f"TARGET = {target_repr}\n"
            f"TARGET_KIND = {target_kind_repr}\n"
            f"SHORTLIST = {shortlist_repr}\n"
            "if DATA.endswith(('.csv',)):\n"
            "    df = pd.read_csv(DATA)\n"
            "elif DATA.endswith(('.tsv', '.txt')):\n"
            "    df = pd.read_csv(DATA, sep='\\t')\n"
            "elif DATA.endswith(('.json',)):\n"
            "    df = pd.read_json(DATA)\n"
            "elif DATA.endswith(('.jsonl', '.ndjson')):\n"
            "    df = pd.read_json(DATA, lines=True)\n"
            "else:\n"
            "    df = pd.read_parquet(DATA)\n"
            "df.head()\n"
        ),
        md_cell("## Preprocessor (from biopsy action plan)"),
        code_cell(pipeline_code),
        md_cell(
            f"## Split\n\n{split_detail}\n\n## CV\n\n{cv_detail}\n"
            + (f"\n{extras_block}\n" if extras_block else "")
        ),
        code_cell(split_code),
        md_cell("## Baseline model on the shortlist"),
        code_cell(baseline_code),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return _json.dumps(notebook, indent=1)
