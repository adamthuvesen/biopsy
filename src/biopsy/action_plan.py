"""Modeling action plan synthesized from a `Profile`.

The action plan is what differentiates `biopsy` from a generic profiler: it
turns ranked findings into concrete, ordered moves a data scientist can act
on before fitting a model — drop, impute, encode, transform, plus a split
and CV recommendation and an optional class-imbalance strategy.

The plan is built from a `Profile` and consumes only its public state
(findings, columns, target_summary, temporal, clusters, target_signals).
HTML, terminal, and sklearn-pipeline codegen all read from the same
`ActionPlan` — there is exactly one place where modeling opinions live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from biopsy.profile import Profile

Severity = Literal["critical", "warning", "info"]
EncodingChoice = Literal["onehot", "ordinal", "target_oof", "hash"]
ImputeChoice = Literal["median", "mean", "mode", "constant_zero", "drop_rows", "model"]
ScalerChoice = Literal["standard", "robust", "log1p", "yeo_johnson", "none"]


@dataclass
class ActionItem:
    """One actionable instruction for one column.

    `evidence` lists the finding titles that justify the action, so a user
    can trace each move back to the underlying signal.
    """

    column: str
    action: str
    reason: str
    severity: Severity
    evidence: list[str] = field(default_factory=list)


@dataclass
class SplitRecommendation:
    kind: Literal["temporal", "stratified", "random"]
    detail: str
    time_column: str | None = None
    cutoff: str | None = None  # for temporal: ISO-ish date string at the train/val boundary
    val_cutoff: str | None = None  # for temporal: ISO-ish date at val/test boundary
    stratify_on: str | None = None


@dataclass
class CVRecommendation:
    kind: Literal["time_series", "stratified_kfold", "kfold", "group_kfold"]
    detail: str
    n_splits: int = 5
    group_column: str | None = None


@dataclass
class ClassStrategy:
    kind: Literal["class_weight", "oversample", "undersample", "focal_loss", "none"]
    detail: str
    positive_rate: float | None = None


@dataclass
class ActionPlan:
    drop: list[ActionItem] = field(default_factory=list)
    review: list[ActionItem] = field(default_factory=list)
    transform: list[ActionItem] = field(default_factory=list)
    encode: list[ActionItem] = field(default_factory=list)
    impute: list[ActionItem] = field(default_factory=list)
    split: SplitRecommendation | None = None
    cv: CVRecommendation | None = None
    class_strategy: ClassStrategy | None = None

    def records(self) -> list[dict[str, Any]]:
        """Flatten into list[dict] for serialization and the API helpers."""
        out: list[dict[str, Any]] = []
        for bucket, items in (
            ("drop", self.drop), ("review", self.review),
            ("transform", self.transform), ("encode", self.encode),
            ("impute", self.impute),
        ):
            for item in items:
                out.append({
                    "bucket": bucket,
                    "column": item.column,
                    "action": item.action,
                    "reason": item.reason,
                    "severity": item.severity,
                    "evidence": list(item.evidence),
                })
        return out


_DROP_TITLE_MARKERS = (
    "is 100% null", "is constant", "is near-constant",
    "looks like an identifier",
)
_REVIEW_CATEGORIES = {"leakage", "temporal", "target"}


def build_action_plan(prof: Profile) -> ActionPlan:
    """Synthesize the modeling action plan from a profile.

    Reads:
      - `prof.findings` for drop / review / transform / impute hints
      - `prof.columns` for null rates, skew, dtype, cardinality
      - `prof.target` and `prof.target_summary` for class-imbalance strategy
      - `prof.time_column` and `prof.temporal` for split recommendation
      - `prof.clusters` for redundancy-aware drops
    """
    drop: dict[str, ActionItem] = {}
    review: dict[str, ActionItem] = {}
    transform: dict[str, ActionItem] = {}
    encode: dict[str, ActionItem] = {}
    impute: dict[str, ActionItem] = {}

    for f in prof.findings:
        if not f.columns:
            continue
        col = f.columns[0]
        # Suppress drop suggestions on the target.
        if col == prof.target and f.category != "leakage":
            continue

        title = f.title.replace("`", "")
        reason = title

        # --- drop bucket: empty, constant, near-constant, ID-shaped,
        #     critical quality ----------------------------------------------
        if any(m in f.title for m in _DROP_TITLE_MARKERS):
            _add(drop, col, ActionItem(
                column=col,
                action="drop",
                reason=reason,
                severity=f.severity,
                evidence=[title],
            ))
            continue
        if f.category == "quality" and f.severity == "critical":
            _add(drop, col, ActionItem(
                column=col,
                action="drop",
                reason=reason,
                severity=f.severity,
                evidence=[title],
            ))
            continue

        # --- review bucket: leakage / temporal / target issues -------------
        if f.category in _REVIEW_CATEGORIES and f.severity in {"critical", "warning"}:
            _add(review, col, ActionItem(
                column=col,
                action="review",
                reason=reason,
                severity=f.severity,
                evidence=[title],
            ))
            continue

        # --- distribution → transform suggestions --------------------------
        if f.category == "distribution":
            stats = prof.columns.get(col)
            is_skewed = (
                stats is not None
                and stats.kind == "numeric"
                and stats.skew is not None
                and abs(stats.skew) > 3
            )
            if is_skewed:
                positive_only = stats.min is not None and stats.min >= 0
                action = "log1p" if positive_only else "yeo_johnson"
            else:
                action = "robust_scaler"
            _add(transform, col, ActionItem(
                column=col,
                action=action,
                reason=reason,
                severity=f.severity,
                evidence=[title],
            ))
            continue

        # --- encoded null sentinels → review (replace before profile) ------
        if (
            f.category == "quality"
            and "encoded nulls" in f.title
        ):
            _add(review, col, ActionItem(
                column=col,
                action="replace_sentinel_with_null",
                reason=reason,
                severity=f.severity,
                evidence=[title],
            ))
            continue

    # --- impute bucket: any non-target column with null_rate in (0, 1) -----
    for stats in prof.columns.values():
        if stats.name == prof.target:
            continue
        if stats.name in drop:
            continue
        if stats.null_rate <= 0 or stats.null_rate >= 1:
            continue
        if stats.kind == "numeric":
            # Heavily skewed → median. Bool-like → mode. Otherwise median.
            action: ImputeChoice = "median"
            if stats.n_unique <= 2:
                action = "mode"
        elif stats.kind in {"text", "bool"}:
            action = "mode"
        else:
            # temporal / other: forward-fill is dataset-specific; leave to user
            continue
        reason = f"{stats.null_rate:.0%} of rows are null"
        _add(impute, stats.name, ActionItem(
            column=stats.name,
            action=action,
            reason=reason,
            severity="warning" if stats.null_rate > 0.1 else "info",
            evidence=[reason],
        ))

    # --- encode bucket: categorical features (skip target, skip dropped) ---
    for stats in prof.columns.values():
        if stats.name == prof.target:
            continue
        if stats.name in drop:
            continue
        if stats.kind not in {"text", "bool"}:
            continue
        n_unique = stats.n_unique
        if n_unique <= 1:
            continue
        nonnull = stats.n - stats.n_null
        unique_rate = n_unique / nonnull if nonnull else 0.0
        if n_unique <= 2:
            choice: EncodingChoice = "ordinal"
            why = "binary categorical"
        elif n_unique <= 20:
            choice = "onehot"
            why = f"low-cardinality ({n_unique} levels)"
        elif unique_rate > 0.5:
            # Near-unique strings — likely free text or IDs. Skip encoding;
            # the drop / suspicious findings should already cover this.
            continue
        else:
            choice = "target_oof"
            why = (
                f"high cardinality ({n_unique:,} levels) — use out-of-fold "
                "target encoding to avoid leakage"
            )
        _add(encode, stats.name, ActionItem(
            column=stats.name,
            action=choice,
            reason=why,
            severity="info",
            evidence=[why],
        ))

    # --- split & cv & class strategy ---------------------------------------
    split, cv = _recommend_split_and_cv(prof)
    class_strategy = _recommend_class_strategy(prof)

    return ActionPlan(
        drop=list(drop.values()),
        review=list(review.values()),
        transform=list(transform.values()),
        encode=list(encode.values()),
        impute=list(impute.values()),
        split=split,
        cv=cv,
        class_strategy=class_strategy,
    )


def _bucket_label(bucket: Any) -> str:
    """TimeBucket dataclass or (label, count) tuple — pull out the label."""
    if hasattr(bucket, "label"):
        return str(bucket.label)
    return str(bucket[0])


def _add(bucket: dict[str, ActionItem], col: str, item: ActionItem) -> None:
    """First-wins per column; merge evidence when an item already exists."""
    existing = bucket.get(col)
    if existing is None:
        bucket[col] = item
        return
    for ev in item.evidence:
        if ev not in existing.evidence:
            existing.evidence.append(ev)


def _recommend_split_and_cv(prof: Profile) -> tuple[SplitRecommendation, CVRecommendation]:
    """Pick split / CV strategy from temporal availability + target shape.

    Priority:
      1. Time column present with ≥3 ordered buckets → temporal split + TimeSeriesSplit.
      2. Imbalanced binary classification target → stratified kfold.
      3. Multiclass or balanced classification → stratified kfold.
      4. Regression / no target → plain kfold.
    """
    time_col = prof.time_column
    temporal = prof.temporal
    target_summary = prof.target_summary

    # 1. Temporal split when there's a usable time column.
    if time_col and temporal is not None and len(temporal.time_buckets) >= 3:
        buckets = temporal.time_buckets
        # cut at 70/15/15 over the ordered buckets
        n = len(buckets)
        cut_train = max(1, int(n * 0.70))
        cut_val = max(cut_train + 1, int(n * 0.85))
        cut_val = min(cut_val, n - 1)
        train_end = _bucket_label(buckets[cut_train - 1])
        val_end = _bucket_label(buckets[cut_val - 1])
        split = SplitRecommendation(
            kind="temporal",
            detail=(
                f"Time-ordered holdout on `{time_col}`: train up to {train_end}, "
                f"validate through {val_end}, test on the remaining period."
            ),
            time_column=time_col,
            cutoff=train_end,
            val_cutoff=val_end,
        )
        cv = CVRecommendation(
            kind="time_series",
            detail=f"`TimeSeriesSplit(n_splits=5)` ordered by `{time_col}`.",
            n_splits=5,
        )
        return split, cv

    # 2 & 3: classification target → stratified.
    if target_summary is not None and target_summary.kind == "classification":
        stratify_on = target_summary.name
        rate = target_summary.positive_rate
        detail = (
            f"Stratified 80/20 holdout on `{stratify_on}` "
            f"(positive rate {rate:.2%})." if rate is not None
            else f"Stratified 80/20 holdout on `{stratify_on}`."
        )
        split = SplitRecommendation(
            kind="stratified",
            detail=detail,
            stratify_on=stratify_on,
        )
        cv = CVRecommendation(
            kind="stratified_kfold",
            detail=f"`StratifiedKFold(n_splits=5, shuffle=True)` on `{stratify_on}`.",
            n_splits=5,
        )
        return split, cv

    # 4: regression / no target.
    detail = "Random 80/20 holdout (no target supplied)." if target_summary is None else (
        f"Random 80/20 holdout on regression target `{target_summary.name}`."
    )
    split = SplitRecommendation(kind="random", detail=detail)
    cv = CVRecommendation(
        kind="kfold",
        detail="`KFold(n_splits=5, shuffle=True)`.",
        n_splits=5,
    )
    return split, cv


def _recommend_class_strategy(prof: Profile) -> ClassStrategy | None:
    summary = prof.target_summary
    if summary is None or summary.kind != "classification":
        return None
    rate = summary.positive_rate
    if rate is None:
        # multiclass — flag if the smallest class is very small
        if summary.min_class_count is not None and summary.min_class_count < 50:
            return ClassStrategy(
                kind="class_weight",
                detail=(
                    f"Smallest class has {summary.min_class_count:,} rows. "
                    "Use `class_weight='balanced'` (sklearn) or sample-weighted loss."
                ),
            )
        return None
    if rate < 0.05:
        return ClassStrategy(
            kind="class_weight",
            detail=(
                f"Severe class imbalance ({rate:.2%} positive). Start with "
                "`class_weight='balanced'`; consider focal loss for "
                "neural / boosted models."
            ),
            positive_rate=rate,
        )
    if rate < 0.20:
        return ClassStrategy(
            kind="class_weight",
            detail=(
                f"Moderate class imbalance ({rate:.2%} positive). Use "
                "`class_weight='balanced'` and stratified CV."
            ),
            positive_rate=rate,
        )
    return None


def categorize_columns(prof: Profile, plan: ActionPlan) -> dict[str, list[str]]:
    """Bucket columns for the sklearn ColumnTransformer codegen.

    Returns dict with keys: `numeric`, `categorical_low`, `categorical_high`,
    `boolean`, `drop`, `passthrough`. The target and dropped columns are
    excluded from feature buckets.
    """
    drop_cols = {item.column for item in plan.drop}
    out: dict[str, list[str]] = {
        "numeric": [],
        "categorical_low": [],
        "categorical_high": [],
        "boolean": [],
        "drop": sorted(drop_cols),
        "passthrough": [],
    }
    encode_kind = {item.column: item.action for item in plan.encode}
    for stats in prof.columns.values():
        if stats.name == prof.target:
            continue
        if stats.name in drop_cols:
            continue
        if stats.kind == "numeric":
            if stats.n_unique <= 2:
                out["boolean"].append(stats.name)
            else:
                out["numeric"].append(stats.name)
        elif stats.kind == "bool":
            out["boolean"].append(stats.name)
        elif stats.kind == "text":
            choice = encode_kind.get(stats.name)
            if choice in {"onehot", "ordinal"}:
                out["categorical_low"].append(stats.name)
            elif choice == "target_oof":
                out["categorical_high"].append(stats.name)
            else:
                # No encoding decision → assume passthrough text (likely
                # free-text or near-unique IDs that the user excludes manually)
                out["passthrough"].append(stats.name)
        else:
            out["passthrough"].append(stats.name)
    for k in ("numeric", "categorical_low", "categorical_high", "boolean", "passthrough"):
        out[k] = sorted(out[k])
    return out


def to_sklearn_pipeline_code(prof: Profile, plan: ActionPlan | None = None) -> str:
    """Emit a runnable Python module that builds a ColumnTransformer.

    The module defines `build_preprocessor() -> ColumnTransformer`. Calling
    code can `fit_transform(df)` on the same dataset profile was run on
    (minus dropped + target columns).
    """
    plan = plan or build_action_plan(prof)
    buckets = categorize_columns(prof, plan)
    impute_kind = {item.column: item.action for item in plan.impute}

    def py_list(items: list[str]) -> str:
        if not items:
            return "[]"
        joined = ", ".join(repr(c) for c in items)
        return f"[{joined}]"

    numeric_impute_strategy = "median"
    categorical_impute_strategy = "most_frequent"
    # If most numeric columns chose mean per the plan, switch; else default to median.
    numeric_modes = [impute_kind.get(c, "median") for c in buckets["numeric"]]
    if numeric_modes and numeric_modes.count("mean") > numeric_modes.count("median"):
        numeric_impute_strategy = "mean"

    target_repr = repr(prof.target) if prof.target else "None"
    dropped_block = "\n".join(
        f"#   - {item.column}: {item.reason}" for item in plan.drop
    ) or "#   (none)"

    split_block = "#   (no split recommendation)"
    if plan.split is not None:
        split_block = f"#   {plan.split.kind}: {plan.split.detail}"
    cv_block = "#   (no cv recommendation)"
    if plan.cv is not None:
        cv_block = f"#   {plan.cv.kind}: {plan.cv.detail}"
    class_block = ""
    if plan.class_strategy is not None:
        class_block = f"\n# Class strategy: {plan.class_strategy.detail}"

    return f'''"""Generated by biopsy. Builds a sklearn ColumnTransformer for {prof.source_name!r}.

Target column: {target_repr}

Split recommendation:
{split_block}

CV recommendation:
{cv_block}{class_block}

Dropped columns (not part of the preprocessor):
{dropped_block}
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = {target_repr}

NUMERIC_COLS = {py_list(buckets["numeric"])}
BOOLEAN_COLS = {py_list(buckets["boolean"])}
CATEGORICAL_LOW_COLS = {py_list(buckets["categorical_low"])}
CATEGORICAL_HIGH_COLS = {py_list(buckets["categorical_high"])}
DROPPED_COLS = {py_list(buckets["drop"])}


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer wired to biopsy\'s suggestions."""
    numeric = Pipeline(steps=[
        ("impute", SimpleImputer(strategy={numeric_impute_strategy!r})),
        ("scale", StandardScaler()),
    ])
    boolean = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    cat_low = Pipeline(steps=[
        (
            "impute",
            SimpleImputer(
                strategy={categorical_impute_strategy!r},
                fill_value="__missing__",
            ),
        ),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # High-cardinality categoricals get ordinal-encoded here as a safe default.
    # Replace with a target encoder (e.g., sklearn.preprocessing.TargetEncoder
    # in 1.3+, or category_encoders) fitted out-of-fold to avoid leakage.
    cat_high = Pipeline(steps=[
        (
            "impute",
            SimpleImputer(
                strategy={categorical_impute_strategy!r},
                fill_value="__missing__",
            ),
        ),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLS),
            ("boolean", boolean, BOOLEAN_COLS),
            ("cat_low", cat_low, CATEGORICAL_LOW_COLS),
            ("cat_high", cat_high, CATEGORICAL_HIGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    pp = build_preprocessor()
    print(pp)
'''
