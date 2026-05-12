"""Generate a synthetic dataset for demoing sketch."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def synthetic_dataframe(n: int = 5000, seed: int = 42) -> dict[str, list]:
    rng = np.random.default_rng(seed)

    age = rng.normal(38, 12, n).clip(18, 90).round().astype(int)
    income = np.exp(rng.normal(10.5, 0.8, n))
    tenure_months = rng.integers(0, 120, n)
    n_logins = rng.poisson(15, n) + (tenure_months // 12)
    plan = rng.choice(["free", "pro", "team", "enterprise"], n, p=[0.55, 0.25, 0.15, 0.05])
    country = rng.choice(["US", "GB", "DE", "FR", "BR", "JP", "IN", "AU"], n)
    referrer_id = rng.integers(1, n + 1, n)
    status = np.full(n, "active", dtype=object)
    constant_col = np.full(n, 42)

    # Sign-up date spans 24 months ending 2026-04-30.
    end_date = datetime(2026, 4, 30)
    start_date = end_date - timedelta(days=730)
    span_days = (end_date - start_date).days
    signup_offsets = rng.integers(0, span_days + 1, n)
    signup_date = [
        (start_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in signup_offsets
    ]
    # is_recent: only the most-recent ~30% of users — chosen to coincide with
    # the 70/30 time-ordered split point in temporal analysis, so a model
    # trained on first-70%-by-time sees zero signal.
    recent_threshold = np.percentile(signup_offsets, 70)
    is_recent = signup_offsets > recent_threshold

    logins_per_month = n_logins / np.maximum(tenure_months, 1)
    churn_logit = (
        -2.0
        + 1.5 * (logins_per_month < 2)
        + 0.8 * (plan == "free")
        + 0.4 * rng.normal(0, 1, n)
        - 0.02 * (age - 38)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (rng.random(n) < churn_prob).astype(int)

    # Direct future-leak: derived from `churned` regardless of time.
    days_since_last_login = np.where(
        churned == 1,
        rng.integers(30, 365, n),
        rng.integers(0, 14, n),
    )

    # TEMPORAL leak: only the recent cohort (top ~30% by signup date) has the
    # column backfilled with values derived from `churned`. Earlier rows hold
    # plain noise. In random-CV, ~30% of training rows carry the signal and the
    # model picks it up; in time-ordered train (first 70% by date), training is
    # pure noise and predictive power collapses on the recent test fold.
    noise = rng.normal(0.5, 0.15, n)
    leaked_signal = np.where(churned == 1, rng.normal(0.9, 0.04, n), rng.normal(0.1, 0.04, n))
    cohort_engagement_v2 = np.where(is_recent, leaked_signal, noise)

    # Add nulls to income
    income = income.astype(object)
    null_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    for i in null_idx:
        income[i] = None

    # Heavy skew + outliers
    revenue = (np.exp(rng.normal(3, 1.5, n)) * (plan != "free")).astype(float)
    revenue[rng.choice(n, 20, replace=False)] *= 50

    return {
        "user_id": list(range(1, n + 1)),
        "signup_date": signup_date,
        "age": age.tolist(),
        "income": income.tolist(),
        "tenure_months": tenure_months.tolist(),
        "n_logins": n_logins.tolist(),
        "plan": plan.tolist(),
        "country": country.tolist(),
        "referrer_id": referrer_id.tolist(),
        "status": status.tolist(),
        "constant_col": constant_col.tolist(),
        "revenue": revenue.tolist(),
        "cohort_engagement_v2": cohort_engagement_v2.tolist(),
        "days_since_last_login": days_since_last_login.tolist(),
        "churned": churned.tolist(),
    }


def write_demo_csv(path: str | Path, n: int = 5000) -> Path:
    import csv
    p = Path(path).expanduser().resolve()
    data = synthetic_dataframe(n)
    cols = list(data.keys())
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([data[c][i] if data[c][i] is not None else "" for c in cols])
    return p
