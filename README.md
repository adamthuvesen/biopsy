# sketch

ML-focused EDA as a Python library and CLI. Point `sketch` at a file or dataframe and get the few modeling risks and opportunities worth acting on: leakage suspects, target signal, drift, nulls, outliers, redundancy, and a ranked feature shortlist.

```python
from sketch import profile

prof = profile(df, target="label")

prof.top_findings()
prof.leakage_suspects()
prof.feature_shortlist(limit=20)
prof.drop_candidates()
```

The CLI uses the same profiling engine:

```bash
sketch profile data.parquet --target label
sketch profile data.parquet --target label --html report.html --open
sketch demo --rows 5000
```

## Why

`ydata-profiling`, SweetViz, and DataPrep are broad report generators. `sketch` is opinionated: it ranks what matters for modeling and keeps the default view small enough to use before training a model.

## What it reports

- **Distributions** — histograms, skew, outliers, near-constant columns
- **Data quality** — null rates, empty columns, identifiers, cardinality
- **Target signal** — MI, PPS, Spearman, AUC, permutation importance
- **Redundancy** — Spearman-distance clustering with a feature shortlist
- **Temporal leakage** — random-vs-time split gaps, monotonicity, target drift
- **Correlations** — Pearson in DuckDB SQL and non-linear MI

## Install

Requires Python 3.11+.

```bash
uv pip install sketch-eda
```

For dataframe frame helpers and in-memory pandas/polars/Arrow tests:

```bash
uv pip install "sketch-eda[dataframe]"
```

For local development:

```bash
git clone <repo>
cd sketch
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Python API

File input:

```python
from sketch import profile

prof = profile(
    "training.parquet",
    target="label",
    time_col="event_time",
    exclude=["row_id"],
    where=["split in train,validation"],
    sample=50_000,
)
```

Dataframe input:

```python
import pandas as pd
from sketch import profile

df = pd.read_parquet("training.parquet")
prof = profile(df, target="label", source_name="training frame")
```

Work with plain Python records:

```python
findings = prof.findings_records()
signals = prof.target_signal_records()
shortlist = prof.shortlist_records()
payload = prof.to_dict()
json_text = prof.to_json()
```

Use notebook-friendly pandas frames when pandas is installed:

```python
prof.findings_frame()
prof.columns_frame()
prof.target_signal_frame()
prof.shortlist_frame()
```

Use ML helpers for modeling decisions:

```python
prof.feature_shortlist(limit=30)      # shortlist from redundancy clusters
prof.leakage_suspects()              # columns suspiciously predictive of target
prof.drop_candidates()               # empty/constant/ID-like/leaky columns
prof.top_findings(category="quality")
```

Supported inputs:

- CSV, TSV, Parquet, JSON paths
- pandas DataFrame
- Polars DataFrame or LazyFrame
- Arrow Table or RecordBatchReader
- DuckDB relations

Pandas, Polars, and Arrow are optional dependencies. `sketch` registers in-memory objects with DuckDB and keeps the CLI dependency set small.

## CLI

```bash
sketch profile <file> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--target / -t COL` | — | Target column for predictive metrics |
| `--time COL` | auto | Time column for temporal leakage |
| `--exclude / -x COL` | — | Drop column from analysis |
| `--filter / -w EXPR` | — | Filter rows before profiling |
| `--sample N` | — | Reservoir-sample N rows after filtering |
| `--shortlist N` | — | Cap the feature shortlist |
| `--cluster-cutoff X` | `0.30` | Cluster cutoff on `1-|rho|` |
| `--html PATH` | — | Write an HTML report |
| `--open` | — | Open the HTML report |
| `--bins N` | `24` | Histogram bin count |

Filter expressions:

```bash
--filter 'segment in train,test'
--filter 'value > 0'
--filter 'event_time is not null'
--filter 'label == positive'
```

Filters are ANDed and applied before sampling.

## Development

```bash
uv run pytest tests/ -q
uv run ruff check src tests
uv run sketch demo --rows 1000 --no-html
```

## License

MIT
