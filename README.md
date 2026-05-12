# sketch

Instant EDA in the terminal. Point it at a CSV or Parquet, get a ranked report of what
actually matters for modeling — distributions, nulls, outliers, non-linear correlations,
suspicious columns.

```bash
sketch profile data.parquet
sketch profile data.parquet --target churn --html report.html
sketch demo   # generate a synthetic dataset and profile it
```

## Why

`ydata-profiling` and friends dump a 200-page report. `sketch` ranks the top things you
should know in one screen and saves the deep-dive for an optional HTML supplement.

## Install (dev)

```bash
uv pip install -e ".[dev]"
```
