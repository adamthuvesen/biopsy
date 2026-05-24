"""Redundancy clustering + interpretability shortlist.

Maps the feature universe onto a small set of clusters where members are highly
rank-correlated (default `|Spearman| ≥ 0.70`), picks a representative per cluster,
and produces a ranked shortlist for stakeholders / modeling baselines.

This module is target-aware when target signals are available: representatives
are chosen by best target-score within each cluster. Without a target, it falls
back to highest within-cluster median absolute correlation (the feature that
"summarizes" its cluster best).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from biopsy.correlations import TargetSignal, _valid_mask
from biopsy.io import Source
from biopsy.matrix import SampleCache, _fetch_object_array
from biopsy.stats import ColumnStats, _quote

DEFAULT_CUTOFF = 0.30  # 1 - |ρ|, so any pair with |ρ| ≥ 0.70 collapses
MIN_CLUSTER_MEMBERS_FOR_DEDUP = 2
WEAK_REP_PPS_THRESHOLD = 0.05  # if target available
WEAK_REP_AUC_THRESHOLD = 0.10  # normalized AUC; raw ~0.55
WEAK_REP_MI_THRESHOLD = 0.05


# --- dataclasses -----------------------------------------------------------


@dataclass
class Cluster:
    cluster_id: int
    members: list[str]
    representative: str
    mean_abs_correlation: float  # average |ρ| among members (1.0 for singletons)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def is_singleton(self) -> bool:
        return len(self.members) <= 1


@dataclass
class ShortlistEntry:
    feature: str
    cluster_id: int
    cluster_size: int
    score: float  # best target-aware score, [0, 1]
    score_method: str  # "pps" | "auc" | "mutual_info" | "no_target"
    is_weak: bool  # flagged as below the weak-rep threshold
    rationale: str  # human-readable per-pick note


@dataclass
class ClusterReport:
    clusters: list[Cluster]
    shortlist: list[ShortlistEntry]
    cutoff: float
    n_features: int = 0
    n_singletons: int = 0

    @property
    def largest_cluster_size(self) -> int:
        return max((c.size for c in self.clusters), default=0)


# --- algorithm -------------------------------------------------------------


def _spearman_distance_matrix(
    src: Source,
    feature_names: list[str],
    max_rows: int = 20_000,
    sample_cache: SampleCache | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return (distance matrix, ordered feature names). Distance = 1 − |ρ|."""
    if len(feature_names) < 2:
        return np.empty((0, 0)), feature_names

    if sample_cache is None:
        quoted = ", ".join(_quote(c) for c in feature_names)
        raw = _fetch_object_array(
            src.con,
            f"SELECT {quoted} FROM data USING SAMPLE {max_rows} ROWS (reservoir, 42)",
            feature_names,
        )
        if raw.size == 0:
            return np.empty((0, 0)), feature_names
    else:
        _cols, raw = sample_cache.fetch(feature_names, max_rows=max_rows)
    if raw.size == 0:
        return np.empty((0, 0)), feature_names
    n = len(feature_names)

    # median-impute missing per column, then rank-transform
    ranks = np.empty((raw.shape[0], n), dtype=np.float64)
    valid_cols: list[int] = []
    for j in range(n):
        col_raw = raw[:, j]
        mask = _valid_mask(col_raw)
        if mask.sum() < 30:
            continue
        clean = np.asarray(col_raw[mask], dtype=np.float64)
        median = float(np.median(clean))
        full = np.empty(raw.shape[0], dtype=np.float64)
        full[mask] = clean
        full[~mask] = median
        ranks[:, j] = rankdata(full, method="average")
        valid_cols.append(j)

    if len(valid_cols) < 2:
        return np.empty((0, 0)), feature_names

    ranks = ranks[:, valid_cols]
    kept = [feature_names[j] for j in valid_cols]
    # Pearson on ranks == Spearman; |corr| → distance.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(ranks.T)
    corr = np.nan_to_num(corr, nan=0.0)
    distance = 1.0 - np.abs(corr)
    # ensure symmetric + zero diagonal (numerical hygiene)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    distance = np.clip(distance, 0.0, 2.0)
    return distance, kept


def _build_clusters(
    distance: np.ndarray,
    feature_names: list[str],
    cutoff: float,
) -> list[Cluster]:
    """Hierarchical clustering with average linkage; cut at `cutoff`."""
    n = len(feature_names)
    if n < 2:
        if not feature_names:
            return []
        return [
            Cluster(
                cluster_id=1,
                members=feature_names,
                representative=feature_names[0],
                mean_abs_correlation=1.0,
            )
        ]
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=cutoff, criterion="distance")
    abs_corr = 1.0 - distance

    buckets: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        buckets.setdefault(int(lbl), []).append(idx)

    clusters: list[Cluster] = []
    # stable ordering: smallest member-name first
    for cid in sorted(
        buckets.keys(),
        key=lambda c: min(feature_names[i] for i in buckets[c]),
    ):
        member_ids = buckets[cid]
        members = [feature_names[i] for i in member_ids]
        # mean |ρ| inside the cluster (1.0 for singletons by definition)
        if len(member_ids) <= 1:
            mean_corr = 1.0
        else:
            sub = abs_corr[np.ix_(member_ids, member_ids)]
            mask = ~np.eye(len(member_ids), dtype=bool)
            mean_corr = float(sub[mask].mean()) if mask.any() else 1.0
        # `representative` is overwritten by _pick_representatives once
        # target signals are available; members[0] is a placeholder.
        clusters.append(
            Cluster(
                cluster_id=len(clusters) + 1,
                members=members,
                representative=members[0],
                mean_abs_correlation=mean_corr,
            )
        )
    return clusters


def _score_for_ranking(sig: TargetSignal | None) -> tuple[float, str]:
    """Best-available signal-strength score for cluster-rep selection.

    Uses PPS / MI / AUC / permutation importance — robust measures of
    "is this feature predictive?". Spearman is excluded because for sparse
    or near-constant columns it can saturate near 1.0 from tied ranks,
    overstating real signal.
    """
    if sig is None:
        return 0.0, "no_target"
    candidates: list[tuple[float, str]] = [(sig.score, "pps"), (sig.mutual_info, "mutual_info")]
    if sig.auc is not None:
        candidates.append((sig.auc, "auc"))
    if sig.perm_importance is not None:
        candidates.append((sig.perm_importance, "perm_importance"))
    return max(candidates, key=lambda t: t[0])


def _pick_representatives(
    clusters: list[Cluster],
    target_signals: dict[str, TargetSignal],
    column_stats: dict[str, ColumnStats],
) -> None:
    """Mutate clusters in place — set `representative` to best-target-score member.

    Without target signals, picks the member with the highest unique-value count
    that isn't suspected of being an identifier (proxy for "most information").
    """
    for cl in clusters:
        if target_signals:
            best = max(
                cl.members,
                key=lambda f: _score_for_ranking(target_signals.get(f))[0],
            )
        else:
            # fallback: prefer the column with the most unique values among
            # non-near-constant numeric columns
            def fallback_score(f: str) -> float:
                s = column_stats.get(f)
                if s is None or s.is_constant or s.is_near_constant:
                    return -1.0
                return float(s.n_unique)

            best = max(cl.members, key=fallback_score)
        cl.representative = best


def _is_weak(score: float, method: str) -> bool:
    if method == "pps":
        return score < WEAK_REP_PPS_THRESHOLD
    if method == "auc":
        return score < WEAK_REP_AUC_THRESHOLD
    if method in {"mutual_info", "perm_importance", "spearman"}:
        return score < WEAK_REP_MI_THRESHOLD
    return True


def _rationale(cl: Cluster, score: float, method: str) -> str:
    if cl.is_singleton:
        if method == "no_target":
            return "Only feature in this cluster; kept for coverage."
        return f"Only feature in this cluster; ranked by {method}={score:.2f}."
    return (
        f"Best target-aware representative among {cl.size} correlated features "
        f"(mean |ρ|={cl.mean_abs_correlation:.2f}); ranked by {method}={score:.2f}."
    )


def cluster_features(
    src: Source,
    stats: dict[str, ColumnStats],
    target: str | None = None,
    target_signals: list[TargetSignal] | None = None,
    cutoff: float = DEFAULT_CUTOFF,
    max_rows: int = 20_000,
    max_shortlist: int | None = None,
    sample_cache: SampleCache | None = None,
) -> ClusterReport:
    """Run clustering + shortlist on numeric features.

    target: feature to exclude from clustering (the prediction target).
    cutoff: 1 − |ρ| threshold. 0.30 ⇔ any pair with |ρ| ≥ 0.70 collapses.
    max_shortlist: cap shortlist length; None = include all clusters.
    """
    eligible = [
        n for n, s in stats.items() if s.kind == "numeric" and not s.is_constant and n != target
    ]
    if len(eligible) < 2:
        return ClusterReport(clusters=[], shortlist=[], cutoff=cutoff, n_features=len(eligible))

    distance, kept = _spearman_distance_matrix(
        src, eligible, max_rows=max_rows, sample_cache=sample_cache
    )
    if distance.size == 0:
        return ClusterReport(clusters=[], shortlist=[], cutoff=cutoff, n_features=len(eligible))

    clusters = _build_clusters(distance, kept, cutoff=cutoff)

    sig_by_feat: dict[str, TargetSignal] = (
        {s.feature: s for s in target_signals} if target_signals else {}
    )
    _pick_representatives(clusters, sig_by_feat, stats)

    # Build shortlist sorted by representative score (descending)
    entries: list[ShortlistEntry] = []
    for cl in clusters:
        sig = sig_by_feat.get(cl.representative)
        score, method = _score_for_ranking(sig)
        weak = _is_weak(score, method) if sig_by_feat else False
        entries.append(
            ShortlistEntry(
                feature=cl.representative,
                cluster_id=cl.cluster_id,
                cluster_size=cl.size,
                score=score,
                score_method=method,
                is_weak=weak,
                rationale=_rationale(cl, score, method),
            )
        )
    entries.sort(key=lambda e: (-e.score, e.feature))

    if max_shortlist is not None:
        entries = entries[:max_shortlist]

    n_singletons = sum(1 for c in clusters if c.is_singleton)
    return ClusterReport(
        clusters=clusters,
        shortlist=entries,
        cutoff=cutoff,
        n_features=len(kept),
        n_singletons=n_singletons,
    )
