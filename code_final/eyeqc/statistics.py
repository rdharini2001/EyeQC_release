"""Small, fast statistical helpers used by the analysis notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


def performance(frame: pd.DataFrame, pass_threshold: float, fail_threshold: float) -> dict:
    data = frame.dropna(subset=["gt_bad", "risk_bad"])
    y = data["gt_bad"].astype(int).to_numpy()
    risk = data["risk_bad"].to_numpy(float)
    failed = risk >= fail_threshold
    passed = risk <= pass_threshold
    tn, fp, fn, tp = confusion_matrix(y, failed, labels=[0, 1]).ravel()
    return {
        "AUROC": roc_auc_score(y, risk),
        "AUPRC": average_precision_score(y, risk),
        "Brier": brier_score_loss(y, risk),
        "FAIL_sensitivity": tp / max(tp + fn, 1),
        "FAIL_specificity": tn / max(tn + fp, 1),
        "poor_capture_nonPASS": np.mean(~passed[y == 1]),
        "PASS_NPV": np.mean(y[passed] == 0) if passed.any() else np.nan,
        "PASS_coverage": passed.mean(),
        "REVIEW_fraction": np.mean((risk > pass_threshold) & (risk < fail_threshold)),
        "FAIL_fraction": failed.mean(),
    }


def _bootstrap_once(
    groups: list[np.ndarray],
    y: np.ndarray,
    risk: np.ndarray,
    pass_threshold: float,
    fail_threshold: float,
    seed: int,
) -> dict | None:
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), len(groups))
    indices = np.concatenate([groups[index] for index in selected])
    if np.unique(y[indices]).size < 2:
        return None
    sample = pd.DataFrame({"gt_bad": y[indices], "risk_bad": risk[indices]})
    return performance(sample, pass_threshold, fail_threshold)


def cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    pass_threshold: float,
    fail_threshold: float,
    *,
    bootstraps: int = 1000,
    workers: int = 4,
    seed: int = 20260822,
) -> dict[str, tuple[float, float]]:
    """Cluster bootstrap without DataFrame concatenation inside the loop."""

    data = frame.dropna(subset=["gt_bad", "risk_bad"]).reset_index(drop=True)
    cluster = data["cluster_id"].astype(str).to_numpy() if "cluster_id" in data else np.arange(len(data)).astype(str)
    y = data["gt_bad"].astype(int).to_numpy()
    risk = data["risk_bad"].to_numpy(float)
    groups = [np.flatnonzero(cluster == value) for value in np.unique(cluster)]
    estimates = Parallel(n_jobs=workers, prefer="threads")(
        delayed(_bootstrap_once)(groups, y, risk, pass_threshold, fail_threshold, seed + index)
        for index in range(bootstraps)
    )
    estimates = [estimate for estimate in estimates if estimate is not None]
    keys = ("AUROC", "AUPRC", "poor_capture_nonPASS", "PASS_NPV", "PASS_coverage")
    return {
        key: tuple(np.quantile([estimate[key] for estimate in estimates], [0.025, 0.975]))
        for key in keys
    }

