# biopsy

![License](https://img.shields.io/github/license/adamthuvesen/biopsy) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

`biopsy` is a Python library and CLI for finding modeling risks before they
reach training. Point it at a CSV, Parquet file, dataframe, or warehouse table
and it returns a short ranked report: temporal leaks, drift, null traps, IDs,
suspicious target signal, redundant features, and a preprocessing plan it can
emit as runnable sklearn code.

The headline check is temporal leakage. `biopsy` compares random-split
predictive power against a time-ordered split, then pairs that with
histogram-based drift. If a feature looks predictive under random CV and
collapses when time is respected, it gets pushed to the top of the report.

## Status and limitations

`biopsy` is early software. Install it from source. It is not on PyPI yet.

The built-in demo uses synthetic data and no secrets. Warehouse support is
read-only, but credentials and network access are still your responsibility.
Treat the findings as review prompts, not automatic data-science decisions.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/adamthuvesen/biopsy.git
cd biopsy
uv venv && source .venv/bin/activate
uv pip install -e .
biopsy demo --rows 1000 --no-html
```

The demo writes a temporary CSV, profiles it, and prints the ranked report:

```text
demo.csv  1,000 rows
biopsy  demo.csv   1,000 rows × 15 cols   target churned

─────────────────────────────────── Findings ───────────────────────────────────
 ■  `days_since_last_login` may leak the target (score=1.00)
    Predictive score against `churned` is suspiciously high. Check whether this
column was computed from the target.
 ■  `cohort_engagement_v2` may leak future information
    Random-CV predictive signal (0.34) but time-ordered split scores near zero
(0.00) with strong drift (KS=0.50).
 ▲  `status` is constant
 ▲  `constant_col` is constant
 ▲  `user_id` looks like an identifier

──────────────────────────── Action plan → churned ─────────────────────────────
  type         column                  action
  drop         status                  drop
  drop         constant_col            drop
  drop         user_id                 drop
  transform    revenue                 log1p
  impute       income                  median
  review       days_since_last_login   review
  review       cohort_engagement_v2    review
```

Other common commands:

```bash
biopsy profile data.parquet --target label
biopsy profile data.parquet --target label --html report.html --pipeline preprocess.py
biopsy compare train.parquet eval.parquet --target label
biopsy doctor data.parquet
```

Python API:

```python
from biopsy import profile

prof = profile("training.parquet", target="label", time_col="event_time")

prof.top_findings()
prof.feature_shortlist(limit=20)
prof.action_plan()
prof.to_sklearn_pipeline_code()
```

## What profilers miss

`ydata-profiling`, SweetViz, and DataPrep describe every column: dozens of
sections of histograms, quantiles, and correlations. They leave one hard
question unanswered: which feature will fail when the split respects time?
Leakage is the clearest case.

`biopsy demo --rows 5000` plants one **temporal leak**: `cohort_engagement_v2` is
backfilled from the outcome for the most recent ~30% of users, so it looks like
a healthy feature. ydata-profiling shows it as 1 of 15 cards with a generic
**High correlation** badge, one of 14 unranked alerts, the same badge it gives
benign pairs. Without a time split, it treats the leak as correlation.

`biopsy` ranks the same column **CRITICAL**, at the top of the report:

```text
 ■  `cohort_engagement_v2` may leak future information
    Predicts target on random CV (0.39) but fails on time-ordered split (0.00).

 Temporal → signup_date
 feature                random→time   drift
 cohort_engagement_v2   0.39 → 0.00    0.53
```

The tell is the collapse: predictive power score (PPS) 0.39 under random CV,
**0.00** under a time-ordered split, plus strong distribution drift. It looks
like a top feature in testing, then contributes nothing in production.

<details>
<summary>Reproduce both numbers (the demo seed is pinned)</summary>

```bash
# identical demo dataset, biopsy's seed is fixed at 42
python -c "from biopsy.demo import write_demo_csv; write_demo_csv('/tmp/demo.csv', n=5000)"
biopsy profile /tmp/demo.csv --target churned

# ydata-profiling: throwaway venv, never added to biopsy's deps
uv venv /tmp/ydata && uv pip install --python /tmp/ydata/bin/python ydata-profiling pandas "setuptools<81"
/tmp/ydata/bin/python -c "import pandas as pd; from ydata_profiling import ProfileReport; ProfileReport(pd.read_csv('/tmp/demo.csv')).to_file('/tmp/ydata.html')"
```

</details>

## What it reports

- Ranked findings: leakage, drift, nulls, outliers, IDs, suspicious date strings,
  near-constant columns, and other modeling risks.
- Target signal: PPS, mutual information, Spearman, AUC, and optional permutation
  importance in `--deep` mode.
- A feature shortlist: one representative from each correlated group, ranked by
  target signal.
- A preprocessing plan: drop, impute, encode, transform, split, CV, and class
  imbalance recommendations.
- Drift reports with schema changes, target movement, and per-column distribution
  changes.
- Optional HTML reports and saved JSON artifacts.

## Optional extras

```bash
uv pip install -e ".[dataframe]"   # pandas, polars, pyarrow
uv pip install -e ".[warehouse]"   # BigQuery, Snowflake, Postgres, object stores
uv pip install -e ".[dev]"         # tests and linting
```

## Python API

Profile a file:

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

Profile a dataframe:

```python
import pandas as pd
from biopsy import profile

df = pd.read_parquet("training.parquet")
prof = profile(df, target="label", source_name="training")
```

Common outputs:

```python
prof.findings_records()
prof.target_signal_records()
prof.shortlist_records()
prof.action_plan_records()
prof.save("profile.json")

prof.findings_frame()       # when pandas is installed
prof.columns_frame()
prof.target_signal_frame()
prof.shortlist_frame()

profile(df, target="label").show()  # notebook HTML
```

Modeling helpers:

```python
plan = prof.action_plan()
plan.drop
plan.impute
plan.encode
plan.transform
plan.split
plan.cv
plan.class_strategy

code = prof.to_sklearn_pipeline_code()
```

Compare two profiles:

```python
from biopsy import compare_profiles

report = compare_profiles(prof_train, prof_eval)
report.schema.added
report.schema.removed
report.target
report.top(10)
```

Diff two profiles:

```python
old = profile("old.parquet", target="label")
new = profile("new.parquet", target="label")

diff = new.diff(old)
diff.appeared
diff.resolved
diff.severity_changed
diff.rank_changed
```

Supported inputs:

- CSV, TSV, Parquet, and JSON paths
- pandas, Polars, Arrow, and DuckDB relation objects
- `s3://`, `gs://`, `https://`, `postgres://`, `bigquery://`, and `snowflake://`
  URIs

Pandas, Polars, Arrow, and warehouse drivers are optional dependencies.

## CLI

```bash
biopsy profile <file-or-uri> [options]
biopsy compare <A> <B> [options]
biopsy diff <a.json> <b.json>
biopsy doctor <file-or-uri>
biopsy init <file>
biopsy notebook <out.ipynb> --file <data> --target <col>
biopsy render <profile.json> --html <out.html>
biopsy demo --rows 5000
```

Useful `profile` options:

| Flag | Meaning |
|---|---|
| `--target, -t COL` | Target column for predictive metrics |
| `--time COL` | Time column for temporal checks; auto-detected when possible |
| `--exclude, -x COL` | Omit a column from analysis; repeatable |
| `--exclude-file PATH` | Omit columns listed one per line |
| `--filter, -w EXPR` | Filter rows before profiling; repeatable |
| `--sample N` | Sample N rows before profiling |
| `--target-sample N` | Sample size for target metrics; default `30000` |
| `--fast / --deep` | `--deep` adds pairwise MI and permutation importance |
| `--max-cols N` | Cap columns in the pairwise MI pass |
| `--shortlist N` | Limit the feature shortlist |
| `--html PATH` | Write an HTML report |
| `--save PATH` | Save profile JSON |
| `--pipeline PATH` | Write sklearn preprocessor code |
| `--config PATH` | Load defaults from TOML |
| `--open` | Open the HTML report |

Filter examples:

```bash
--filter 'segment in train,test'
--filter 'value > 0'
--filter 'event_time is not null'
--filter 'label == positive'
```

Config files remove repeated flags:

```toml
target = "target"
time = "snapshot_date"
filter = ["segment in A,B,C"]
exclude = ["account_key", "segment"]
fast = true

[profiles.deep]
fast = false
plotly_cdn = true
```

```bash
biopsy profile data.parquet --config biopsy.toml
biopsy profile data.parquet --config biopsy.toml --profile-name deep --html report.html
```

Generate a starter config:

```bash
biopsy init data.parquet
```

## Warehouse Sources

Pass a URI anywhere a file path is accepted:

```bash
biopsy profile s3://my-bucket/events.parquet --target conversion
biopsy profile postgres://localhost/sales?table=public.orders --target shipped
biopsy profile bigquery://my-project/analytics.events --target conversion --sample 50000
biopsy profile snowflake://my-acct/SALES.PUBLIC.ORDERS --target shipped --sample 50000
biopsy compare \
  postgres://localhost/sales?table=public.train \
  postgres://localhost/sales?table=public.eval \
  --target shipped
biopsy doctor snowflake://my-acct/SALES.PUBLIC.ORDERS
```

Warehouse reads are pull-only. `doctor` uses schema discovery and does not pull
row data. For `profile`, filters and limits are pushed down for BigQuery and
Snowflake; object-store and Postgres reads go through DuckDB. `--sample N`
becomes `LIMIT N` for warehouse sources, so use `--filter` when the first N rows
would be biased.

Credentials come from environment variables and are not stored in profiles or
printed in progress output.

| Scheme | Env vars |
|---|---|
| `s3://`, `s3a://` | Optional `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`, `AWS_DEFAULT_REGION` |
| `gs://`, `gcs://` | Optional `GOOGLE_APPLICATION_CREDENTIALS` |
| `https://`, `http://` | Optional `BIOPSY_HTTPS_BEARER` |
| `postgres://`, `postgresql://` | Optional libpq vars: `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `PGPORT`, `PGSSLMODE` |
| `bigquery://` | Required `GOOGLE_APPLICATION_CREDENTIALS`; optional `BIGQUERY_PROJECT` |
| `snowflake://` | Required `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, and either `SNOWFLAKE_PRIVATE_KEY_PATH` or `SNOWFLAKE_PASSWORD`; optional `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA` |

Use a prefix for another credential set:

```bash
biopsy profile s3://staging-bucket/events.parquet --credentials-env STAGING
biopsy compare \
  postgres://prod/sales?table=public.train \
  postgres://prod/sales?table=public.eval \
  --credentials-env STAGING
# reads STAGING_AWS_ACCESS_KEY_ID, STAGING_AWS_SECRET_ACCESS_KEY, ...
```

Install only the backends you need:

```bash
uv pip install -e ".[object-store]"
uv pip install -e ".[postgres]"
uv pip install -e ".[bigquery]"
uv pip install -e ".[snowflake]"
uv pip install -e ".[warehouse]"
```

## Development

```bash
uv run pytest tests/ -q
uv run ruff check src tests
uv run mypy
uv run biopsy demo --rows 1000
```

## License

MIT
