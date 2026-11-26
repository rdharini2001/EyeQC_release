"""Local probability recalibration, and the operating metrics used to judge it.

Given a few locally labelled photographs, a two-parameter monotone logistic function is fitted to the frozen EyeQC risk::

    risk_local = sigmoid(a + b * logit(risk)),   b >= 0

with a quadratic penalty towards the identity (a = 0, b = 1).  Features, classifier and thresholds are not changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from .contract import FAIL_THRESHOLD, PASS_THRESHOLD

CLIP = 1e-6


@dataclass(frozen=True)
class Recalibration:
    intercept: float
    slope: float

    def __call__(self, risk) -> np.ndarray:
        risk = np.clip(np.asarray(risk, float), CLIP, 1 - CLIP)
        return expit(self.intercept + self.slope * logit(risk))


def fit_recalibration(risk, is_poor) -> Recalibration:
    """Fit on locally labelled photographs. ``is_poor`` is 1 for poor-quality reference labels, 0 otherwise."""
    risk = np.asarray(risk, float)
    labels = np.asarray(is_poor, float)
    if risk.shape != labels.shape or risk.size == 0:
        raise ValueError("risk and is_poor must be non-empty arrays of equal length")
    if not np.isin(labels, (0.0, 1.0)).all() or np.unique(labels).size < 2:
        raise ValueError("is_poor must contain both 0 and 1")
    x = logit(np.clip(risk, CLIP, 1 - CLIP))

    def objective(params):
        a, b = params
        z = a + b * x
        value = np.sum(np.logaddexp(0, z) - labels * z) + 0.5 * (a**2 + (b - 1) ** 2)
        error = expit(z) - labels
        return value, np.array([error.sum() + a, (error * x).sum() + b - 1])

    fit = minimize(objective, np.array([0.0, 1.0]), jac=True, bounds=[(None, None), (0, None)], method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"recalibration did not converge: {fit.message}")
    return Recalibration(float(fit.x[0]), float(fit.x[1]))


def local_pass_threshold(risk, is_poor, sensitivity: float = 0.90, fail_threshold: float = FAIL_THRESHOLD) -> float:
    """Exploratory local PASS cut-off: just below the risk that keeps ``sensitivity`` of locally labelled poor photographs
    out of PASS, never above the FAIL threshold."""
    risk = np.asarray(risk, float)
    poor = risk[np.asarray(is_poor, int) == 1]
    if poor.size == 0:
        raise ValueError("The labelled set contains no poor photograph.")
    cut = np.nextafter(np.quantile(poor, round(1 - sensitivity, 10), method="lower"), -np.inf)
    return float(min(cut, fail_threshold))


def labelled_participants(groups: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of a labelled set of at most ``budget`` photographs made of whole participants, drawn at random."""
    unique = np.unique(groups)
    sizes = dict(zip(*np.unique(groups, return_counts=True)))
    picks, count = [], 0
    for group in rng.permutation(unique):
        if count + sizes[group] <= budget:
            picks.append(group)
            count += sizes[group]
        if count == budget:
            break
    return np.isin(groups, picks)


def operating_metrics(is_poor, risk, pass_threshold: float = PASS_THRESHOLD, fail_threshold: float = FAIL_THRESHOLD) -> dict:
    """Calibration and fixed-threshold metrics (expected calibration error over ten equal-width bins)."""
    y = np.asarray(is_poor, int)
    p = np.asarray(risk, float)
    passed, failed = p <= pass_threshold, p >= fail_threshold
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum((bins == j).mean() * abs((p[bins == j] - y[bins == j]).mean()) for j in range(10) if (bins == j).any())

    def share(numerator, denominator):
        return float(numerator.sum() / denominator.sum()) if denominator.sum() else float("nan")

    return {
        "n": int(y.size),
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size > 1 else float("nan"),
        "ECE": float(ece),
        "Brier": float(np.mean((p - y) ** 2)),
        "false_pass_rate": share(passed & (y == 1), y == 1),
        "pass_npv": share(passed & (y == 0), passed),
        "fail_sensitivity": share(failed & (y == 1), y == 1),
        "pass_share": float(passed.mean()),
        "review_share": float((~passed & ~failed).mean()),
    }
