"""Agreement between graders, and between EyeQC and graders, on the ordered scale FAIL = 0, REVIEW = 1, PASS = 2.

These are the statistics of the reader study: quadratic-weighted Cohen's kappa, Fleiss' kappa, ordinal Krippendorff's
alpha, ICC(3,1) / ICC(3,k), Spearman correlation and per-grader AUROC of the EyeQC risk, and the paired-bootstrap
comparison of EyeQC-grader agreement with grader-grader agreement (equivalence margin +/- 0.15).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

CODES = {"FAIL": 0, "REVIEW": 1, "PASS": 2}
LABELS = (0, 1, 2)
EQUIVALENCE_MARGIN = 0.15


def to_codes(values) -> np.ndarray:
    """Accept PASS / REVIEW / FAIL (any case) or 0 / 1 / 2."""
    series = pd.Series(values)
    if series.dtype.kind in "iuf":
        codes = series.to_numpy(float)
    else:
        text = series.astype(str).str.strip()
        codes = text.str.upper().map(CODES).fillna(pd.to_numeric(text, errors="coerce")).to_numpy(float)
    if not np.isin(codes, LABELS).all():
        raise ValueError("Grades must be PASS, REVIEW or FAIL (or 2, 1, 0) with no missing values.")
    return codes.astype(int)


def quadratic_kappa(a, b) -> float:
    return float(cohen_kappa_score(a, b, labels=list(LABELS), weights="quadratic"))


def _category_counts(ratings: np.ndarray) -> np.ndarray:
    return np.stack([(ratings == label).sum(axis=1) for label in LABELS], axis=1).astype(float)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """``ratings``: photographs x graders, integer categories."""
    counts = _category_counts(ratings)
    graders = ratings.shape[1]
    observed = ((counts * (counts - 1)).sum(axis=1) / (graders * (graders - 1))).mean()
    shares = counts.sum(axis=0) / counts.sum()
    expected = float((shares**2).sum())
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def krippendorff_ordinal_alpha(ratings: np.ndarray) -> float:
    """Krippendorff's alpha with ordinal distances, for complete photographs x graders ratings."""
    counts = _category_counts(ratings)
    graders = ratings.shape[1]
    pairs = counts[:, :, None] * counts[:, None, :]
    pairs[:, range(len(LABELS)), range(len(LABELS))] -= counts
    coincidences = pairs.sum(axis=0) / (graders - 1)
    totals = coincidences.sum(axis=1)
    distance = np.zeros((len(LABELS), len(LABELS)))
    for c in range(len(LABELS)):
        for k in range(len(LABELS)):
            low, high = min(c, k), max(c, k)
            distance[c, k] = (totals[low:high + 1].sum() - (totals[low] + totals[high]) / 2) ** 2 if c != k else 0.0
    observed = float((coincidences * distance).sum())
    expected = float((np.outer(totals, totals) * distance).sum() / (totals.sum() - 1))
    return float(1 - observed / expected) if expected > 0 else float("nan")


def icc3(ratings: np.ndarray) -> tuple[float, float]:
    """(ICC(3,1), ICC(3,k)): two-way mixed-effects model with graders treated as fixed."""
    x = np.asarray(ratings, float)
    n, k = x.shape
    grand = x.mean()
    between = k * ((x.mean(axis=1) - grand) ** 2).sum()
    graders = n * ((x.mean(axis=0) - grand) ** 2).sum()
    error = ((x - grand) ** 2).sum() - between - graders
    ms_between, ms_error = between / (n - 1), error / ((n - 1) * (k - 1))
    single = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)
    return float(single), float((ms_between - ms_error) / ms_between)


def _resamples(groups: np.ndarray, count: int, rng: np.random.Generator):
    """Row indices of bootstrap samples that draw whole participants."""
    members = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    for _ in range(count):
        yield np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])


def reader_report(grades: pd.DataFrame, eyeqc, risk=None, participants=None, *, bootstraps: int = 2000,
                  margin: float = EQUIVALENCE_MARGIN, seed: int = 20261011) -> dict:
    """Full agreement report.

    ``grades``: one column per grader (two or more), one row per photograph.  ``eyeqc``: EyeQC categories for the same
    photographs.  ``risk``: optional EyeQC poor-quality risk.  ``participants``: optional identifiers for resampling.
    """
    if grades.shape[1] < 2:
        raise ValueError("At least two grader columns are needed.")
    names = [str(name) for name in grades.columns]
    human = np.stack([to_codes(grades[column]) for column in grades.columns], axis=1)
    model = to_codes(eyeqc)
    if len(model) != len(human) or len(human) < 10:
        raise ValueError("Need at least 10 photographs with a grade from every grader and from EyeQC.")
    groups = np.arange(len(human)) if participants is None else np.asarray(participants).astype(str)
    everyone = np.column_stack([human, model])
    all_names = names + ["EyeQC"]
    matrix = pd.DataFrame([[quadratic_kappa(everyone[:, i], everyone[:, j]) for j in range(len(all_names))]
                           for i in range(len(all_names))], index=all_names, columns=all_names)

    rng = np.random.default_rng(seed)
    samples = list(_resamples(groups, bootstraps, rng))
    graders = []
    for i, name in enumerate(names):
        others = [j for j in range(len(names)) if j != i]

        def delta(rows, i=i, others=others):
            return quadratic_kappa(human[rows, i], model[rows]) - np.mean([quadratic_kappa(human[rows, i], human[rows, j]) for j in others])

        values = np.array([delta(rows) for rows in samples])
        low, high = np.nanpercentile(values, [2.5, 97.5])
        row = {"grader": name, "photographs": len(human), "kappa_with_EyeQC": quadratic_kappa(human[:, i], model),
               "mean_kappa_with_other_graders": float(np.mean([quadratic_kappa(human[:, i], human[:, j]) for j in others])),
               "difference": float(delta(np.arange(len(human)))), "ci_low": float(low), "ci_high": float(high),
               "equivalent_within_margin": bool(low >= -margin and high <= margin)}
        if risk is not None:
            scores = np.asarray(risk, float)
            failed = human[:, i] == CODES["FAIL"]
            row["spearman_risk_vs_category"] = float(-spearmanr(scores, human[:, i]).statistic)
            row["auroc_for_grader_FAIL"] = float(roc_auc_score(failed, scores)) if failed.any() and (~failed).any() else float("nan")
        graders.append(row)

    group_rows = []
    for label, ratings in (("graders", human), ("graders + EyeQC", everyone)):
        single, average = icc3(ratings)
        group_rows.append({"group": label, "raters": ratings.shape[1], "fleiss_kappa": fleiss_kappa(ratings),
                           "krippendorff_alpha_ordinal": krippendorff_ordinal_alpha(ratings), "icc_3_1": single, "icc_3_k": average,
                           "unanimous_share": float((ratings == ratings[:, [0]]).all(axis=1).mean())})
    return {"kappa_matrix": matrix, "per_grader": pd.DataFrame(graders), "group": pd.DataFrame(group_rows)}
