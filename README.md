# biopsy

ML-focused EDA. Rank features by predictive signal (PPS, MI, AUC, permutation importance), pick a non-redundant modeling shortlist via correlation clustering, and emit a runnable sklearn preprocessor. Catches leakage, drift, and quality issues along the way.

```python
from biopsy import profile

prof = profile(df, target="label")

prof.feature_shortlist(limit=20)   # non-redundant features, ranked by signal
prof.target_signal_records()       # PPS / MI / ρ / AUC / perm importance per feature
prof.top_findings()                # findings ranked by severity
prof.action_plan()                 # drop / impute / encode / transform / split / cv
prof.to_sklearn_pipeline_code()    # runnable ColumnTransformer module
```

CLI:

```bash
biopsy profile data.parquet --target label
biopsy profile data.parquet --target label --html report.html --open
biopsy profile data.parquet --target label --pipeline preprocess.py
biopsy compare train.parquet eval.parquet --target label
biopsy diff old_profile.json new_profile.json
biopsy doctor data.parquet                 # schema + candidate target/time columns
biopsy demo --rows 5000
```

## Why

`ydata-profiling`, SweetViz, and DataPrep generate broad descriptive reports. `biopsy` ranks features by predictive signal and emits a runnable preprocessing pipeline — short enough to read before training a model, actionable enough to skip the boilerplate.

## What it reports

- **Target signal** — every feature ranked by MI, PPS, Spearman, raw AUC, AUC lift; optional permutation importance, bootstrap CIs, and multi-seed PPS stability in `--deep` mode
- **Feature shortlist** — Spearman-distance clustering picks one representative per correlated group, ranked by signal — the non-redundant features worth modeling with
- **Correlations** — Pearson in DuckDB SQL plus a non-linear MI pass that catches what Pearson misses
- **Action plan** — drop / impute / encode / transform buckets, a split + CV + class-imbalance recommendation, and a runnable sklearn `ColumnTransformer` module
- **Distributions** — histograms, skew, outliers, near-constant columns
- **Data quality** — null rates, empty columns, identifiers, cardinality, encoded null sentinels, date-strings, bool-like ints, free-text
- **Leakage detection** — flags features suspiciously predictive of the target (random-vs-time split gaps, monotonicity, post-event signals)
- **Drift** (`biopsy compare`) — KS, Wasserstein, PSI on numerics; chi-square + JS divergence on categoricals; schema + target deltas

## Install

Requires Python 3.11+. Not on PyPI yet — install from source:

```bash
git clone <repo>
cd biopsy
uv venv && source .venv/bin/activate
uv pip install -e .                  # CLI + library
uv pip install -e ".[dataframe]"     # + pandas / polars / pyarrow helpers
uv pip install -e ".[dev]"           # + test/lint tooling
```

## Python API

File input:

```python
from biopsy import profile

prof = profile(
    "training.parquet",
    target="label",
    time_col="event_time",
    exclude=["row_id"],
    where=["split in train,validation"],
    sample=50_000,
)
```

DataFrame input:

```python
import pandas as pd
from biopsy import profile

df = pd.read_parquet("training.parquet")
prof = profile(df, target="label", source_name="training frame")
```

Plain Python records and serialization:

```python
findings = prof.findings_records()
signals = prof.target_signal_records()
shortlist = prof.shortlist_records()
plan = prof.action_plan_records()
payload = prof.to_dict()
prof.save("profile.json")
```

Pandas frames (when pandas is installed):

```python
prof.findings_frame()
prof.columns_frame()
prof.target_signal_frame()
prof.shortlist_frame()
```

In notebooks, the profile renders itself as HTML:

```python
profile(df, target="label").show()
```

Modeling helpers:

```python
prof.feature_shortlist(limit=30)      # cluster representatives, ranked
prof.leakage_suspects()               # columns suspiciously predictive of target
prof.drop_candidates()                # empty/constant/ID-like/leaky columns
prof.top_findings(category="quality")

plan = prof.action_plan()
plan.drop                             # list[ActionItem]
plan.impute                           # list[ActionItem]
plan.encode                           # list[ActionItem]
plan.transform                        # list[ActionItem]
plan.split                            # SplitRecommendation (temporal / stratified / random)
plan.cv                               # CVRecommendation
plan.class_strategy                   # ClassStrategy or None

# Generate a runnable sklearn ColumnTransformer module:
Path("preprocess.py").write_text(prof.to_sklearn_pipeline_code())
```

Drift between two profiles or two datasets:

```python
from biopsy import compare_profiles

report = compare_profiles(prof_train, prof_eval)
report.schema.added, report.schema.removed
report.target.detail                  # target-rate or target-mean delta
report.top(10)                        # most-drifted features by KS / PSI / JS
```

Finding-level diff between two saved profiles:

```python
diff = prof_v2.diff(prof_v1)
diff.appeared, diff.resolved, diff.severity_changed, diff.rank_changed
```

Supported inputs:

- CSV, TSV, Parquet, JSON paths
- pandas DataFrame
- Polars DataFrame or LazyFrame
- Arrow Table or RecordBatchReader
- DuckDB relations
- Object-store URIs: `s3://bucket/key.parquet`, `https://host/file.parquet`, `gs://bucket/key.parquet`

Pandas, Polars, and Arrow are optional dependencies. `biopsy` registers in-memory objects with DuckDB and keeps the core dependency set small.

## Warehouse sources

Biopsy profiles data where it lives — no need to export to Parquet first. Pass a URI anywhere a file path is accepted:

```bash
biopsy profile s3://my-bucket/events.parquet --target conversion
biopsy profile postgres://localhost/sales?table=public.orders --target shipped
biopsy profile bigquery://my-project/analytics.events --target conversion --sample 50000
biopsy profile snowflake://my-acct/SALES.PUBLIC.ORDERS --target shipped --sample 50000
biopsy profile https://host/data.csv --sample 20000

# `doctor` runs schema-only — no row data transferred for any warehouse source.
biopsy doctor s3://my-bucket/big.parquet              # reads the Parquet footer
biopsy doctor postgres://host/db?table=public.events  # information_schema + pg_class
biopsy doctor bigquery://my-project/analytics.events  # information_schema + __TABLES__
biopsy doctor snowflake://my-acct/SALES.PUBLIC.ORDERS # information_schema
```

Read-only / pull-only by construction. Biopsy never writes back to a warehouse — adapters only issue `SELECT`s, enforced by a lint test in CI. Per-backend safety nets layer on top: Postgres uses `ATTACH … READ_ONLY` + session `default_transaction_read_only=on`; BigQuery/Snowflake adapters issue only `SELECT` via their vendor clients.

The `--sample N` flag becomes `LIMIT N` against warehouse sources (head-of-table, not random); use `--filter` for stratification. For BigQuery and Snowflake, `--filter` predicates are pushed into the remote `SELECT` so the table never transfers in full. BigQuery runs a dry-run estimate first and warns above ~5 GB scanned; Snowflake warns when the Arrow materialization would exceed ~5×10⁸ cells.

Auth comes from environment variables. Credentials never appear on the command line, in saved profiles, or in progress output:

| Scheme | Required env vars |
|---|---|
| `s3://`, `s3a://` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, `AWS_REGION` |
| `gs://`, `gcs://` | `GOOGLE_APPLICATION_CREDENTIALS` (path to service-account JSON) |
| `https://`, `http://` | none required (public). Optional `BIOPSY_HTTPS_BEARER` for bearer auth (future) |
| `postgres://`, `postgresql://` | libpq vars: `PGUSER`, `PGPASSWORD`, `PGHOST`, `PGPORT`, `PGDATABASE`, optional `PGSSLMODE`. URI host/port/db override env. |
| `bigquery://` | `GOOGLE_APPLICATION_CREDENTIALS`, optional `BIGQUERY_PROJECT` (URI host overrides). |
| `snowflake://` | `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, plus `SNOWFLAKE_PRIVATE_KEY_PATH` (key-pair) **or** `SNOWFLAKE_PASSWORD`. Optional: `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`. |

Use a different credential set with `--credentials-env PREFIX`:

```bash
# Reads STAGING_AWS_ACCESS_KEY_ID etc. instead of AWS_ACCESS_KEY_ID.
biopsy profile s3://staging-bucket/events.parquet --credentials-env STAGING
```

Install backends as needed (none are required for the default install):

```bash
pip install 'biopsy[object-store]'   # S3, HTTPS, GCS (no extra Python deps; uses DuckDB httpfs)
pip install 'biopsy[postgres]'       # Postgres / Redshift (no extra Python deps; uses DuckDB postgres extension)
pip install 'biopsy[bigquery]'       # BigQuery via google-cloud-bigquery
pip install 'biopsy[snowflake]'      # Snowflake via snowflake-connector-python
pip install 'biopsy[warehouse]'      # everything
```

## CLI

```bash
biopsy profile <file-or-uri> [options]
biopsy compare <A> <B> [options]              # data files, URIs, or saved JSONs
biopsy diff <a.json> <b.json>
biopsy doctor <file-or-uri>                   # schema + candidate target/time columns
biopsy init <file>                            # write a starter biopsy.toml
biopsy notebook <out.ipynb> --file <data> --target <col>
biopsy render <profile.json> --html <out.html>
```

### `biopsy profile` flags

| Flag | Default | Description |
|---|---|---|
| `--target / -t COL` | — | Target column for predictive metrics |
| `--time COL` | auto | Time column for temporal leakage |
| `--exclude / -x COL` | — | Drop column from analysis (repeatable) |
| `--exclude-file PATH` | — | Drop columns listed one-per-line |
| `--ignore-missing-exclude` | off | Skip absent excluded columns |
| `--filter / -w EXPR` | — | Filter rows before profiling (repeatable, ANDed) |
| `--sample N` | — | Reservoir-sample N rows after filtering |
| `--target-sample N` | `30000` | Target-metric sample size; stratified for rare classification targets |
| `--fast / --deep` | `--fast` | `--deep` adds pairwise MI and target permutation |
| `--max-cols N` | — | Cap columns in the pairwise MI pass (wide-dataset speedup) |
| `--all-columns` | off | Print the full terminal column table |
| `--shortlist N` | — | Cap the feature shortlist |
| `--cluster-cutoff X` | `0.30` | Cluster cutoff on `1 - |ρ|` |
| `--html PATH` | — | Write an HTML report |
| `--save PATH` | — | Save the reusable profile JSON artifact |
| `--pipeline PATH` | — | Write a sklearn ColumnTransformer module from the action plan |
| `--plotly-cdn` | off | Load Plotly from CDN instead of embedding |
| `--config PATH` | — | TOML config with defaults; pair with `--profile-name` for `[profiles.NAME]` |
| `--bins N` | `24` | Histogram bin count |
| `--open` | — | Open the HTML report |

Filter expressions:

```bash
--filter 'segment in train,test'
--filter 'value > 0'
--filter 'event_time is not null'
--filter 'label == positive'
```

Config files keep repeated project choices out of the shell:

```toml
target = "target"
time = "snapshot_date"
filter = ["segment in A,B,C"]
exclude = ["account_key", "segment", "score_column"]
ignore_missing_exclude = true
fast = true

[profiles.deep]
fast = false
plotly_cdn = true
```

```bash
biopsy profile data.parquet --config biopsy.toml
biopsy profile data.parquet --config biopsy.toml --profile-name deep --html report.html
```

Unknown TOML keys are rejected with a did-you-mean suggestion.

Generate a starter config from a real file:

```bash
biopsy init data.parquet
biopsy profile data.parquet --config biopsy.toml --save profile.json
biopsy render profile.json --html report.html
```

### `biopsy compare`

```bash
biopsy compare train.parquet eval.parquet --target label --time event_time --html drift.html
biopsy compare profile_v1.json profile_v2.json --save compare.json
```

Accepts either two data files or two saved profile JSONs. Reports schema diff, target delta, and per-column drift (KS / Wasserstein / PSI for numerics; chi-square + JS divergence for categoricals). HTML output includes per-feature side-by-side distributions.

### `biopsy diff`

```bash
biopsy diff old_profile.json new_profile.json
```

Finding-level changes between two saved profiles: appeared / resolved findings, severity changes, schema changes, target-signal rank shifts.

### `biopsy doctor`

```bash
biopsy doctor data.parquet
```

Schema preview with candidate target columns, candidate time columns, and per-column "looks like" hints (identifier, boolean, low-card categorical, high-null). Sub-second on most datasets; does not run the full profile.

### `biopsy notebook`

```bash
biopsy notebook starter.ipynb --file data.parquet --target label
```

Writes a notebook scaffold using the action plan's preprocessor, shortlist, split, and CV recommendation as a starting point.

## Development

```bash
uv run pytest tests/ -q
uv run ruff check src tests
uv run biopsy demo --rows 1000 --no-html
```

## License

MIT
