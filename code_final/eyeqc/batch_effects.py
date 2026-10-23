"""Nested batch-effect decomposition, ComBat-style correction, and pathology
quality-disentanglement diagnostics.

This module is deliberately separate from :mod:`eyeqc.experiments`, which
keeps the fast, ad hoc robust location-scale harmonizer used for the GUI's
original two-site comparison. Everything here answers three sharper
questions with an established statistical method rather than a heuristic:

1. How much of the ten-axis quality variance sits *between* cohorts
   (inter-batch, usually institution/country/camera) versus *between
   sub-batches nested inside a cohort* (intra-batch: a screening split, a
   second academic centre, a disease subgroup collected on the same
   protocol) versus ordinary image-to-image noise? :func:`variance_decomposition`.
2. If two or more batches differ, what does a principled empirical-Bayes
   batch correction (ComBat; Johnson, Li & Rabinovic 2007) look like, and
   does it actually reduce batch separability without being accepted blindly?
   :func:`fit_combat` / :func:`apply_combat` / :func:`governed_combat_correction`.
3. Is any quality axis systematically confounded with disease status rather
   than acquisition technique, so that an automated FAIL route would
   disproportionately reject diseased eyes for the wrong reason?
   :func:`pathology_quality_disentanglement`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import FEATURES
from .experiments import domain_separability


# ---------------------------------------------------------------------------
# 1. Nested (inter-batch / intra-batch) variance decomposition
# ---------------------------------------------------------------------------


def variance_decomposition(
    frame: pd.DataFrame,
    features: tuple[str, ...] = FEATURES,
    outer: str = "cohort",
    inner: str | None = "subcohort",
) -> pd.DataFrame:
    """Nested-ANOVA sum-of-squares decomposition of each axis's variance.

    Splits the total variance of every quality axis into three additive
    fractions: between ``outer`` batches (for example cohort/institution),
    between ``inner`` sub-batches nested inside each outer batch (for
    example a second academic centre or a disease subgroup collected on the
    same protocol), and residual image-to-image variance. A high
    inter-batch fraction on an axis means pooling that axis across cohorts
    is unsafe without correction; a high intra-batch fraction means even a
    single cohort is not internally homogeneous.
    """
    data = frame.dropna(subset=[outer, *([inner] if inner else [])] + list(features)).copy()
    if data.empty:
        raise ValueError("No complete rows for variance decomposition")

    rows = []
    for feature in features:
        values = data[feature].to_numpy(float)
        grand_mean = values.mean()
        ss_total = float(np.sum((values - grand_mean) ** 2))

        ss_outer = 0.0
        ss_inner = 0.0
        for _, outer_group in data.groupby(outer):
            outer_values = outer_group[feature].to_numpy(float)
            outer_mean = outer_values.mean()
            ss_outer += len(outer_values) * (outer_mean - grand_mean) ** 2
            if inner is not None:
                for _, inner_group in outer_group.groupby(inner):
                    inner_values = inner_group[feature].to_numpy(float)
                    ss_inner += len(inner_values) * (inner_values.mean() - outer_mean) ** 2

        ss_residual = max(ss_total - ss_outer - ss_inner, 0.0)
        denom = max(ss_total, 1e-12)
        rows.append(
            {
                "feature": feature,
                "ss_total": ss_total,
                "inter_batch_fraction": ss_outer / denom,
                "intra_batch_fraction": ss_inner / denom,
                "residual_fraction": ss_residual / denom,
                "n_outer_batches": data[outer].nunique(),
                "n_inner_batches": data[inner].nunique() if inner is not None else 0,
                "n_images": len(data),
            }
        )
    return pd.DataFrame(rows).sort_values("inter_batch_fraction", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. ComBat-style empirical-Bayes batch correction
# ---------------------------------------------------------------------------


@dataclass
class ComBatContract:
    """Fitted ComBat parameters; enough to reproduce the correction exactly."""

    features: tuple[str, ...]
    batch_column: str
    grand_mean: pd.Series
    pooled_std: pd.Series
    batches: list[str]
    gamma_star: pd.DataFrame  # index=batch, columns=features
    delta_star: pd.DataFrame  # index=batch, columns=features
    n_per_batch: pd.Series


def _it_sol(z: np.ndarray, gamma_hat: np.ndarray, delta_hat: np.ndarray, gamma_bar: float, tau2: float, a: float, b: float, conv: float = 1e-4, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Iterative empirical-Bayes solve for one batch, vectorized across features.

    ``z`` is (n_samples, n_features) standardized data for one batch. This is
    the standard ComBat fixed-point iteration (Johnson et al. 2007), applied
    here with "genes" reinterpreted as EyeQC's ten quality axes: strength is
    borrowed across axes within a batch rather than across thousands of genes,
    which shrinks unstable estimates from small batches toward the batch's own
    axis-average while leaving large, well-estimated batches nearly untouched.
    """
    n = z.shape[0]
    gamma_old, delta_old = gamma_hat.copy(), delta_hat.copy()
    for _ in range(max_iter):
        gamma_new = (n * tau2 * gamma_hat + delta_old * gamma_bar) / (n * tau2 + delta_old)
        sum_sq = np.sum((z - gamma_new[None, :]) ** 2, axis=0)
        delta_new = (0.5 * sum_sq + b) / (n / 2.0 + a - 1.0)
        delta_new = np.clip(delta_new, 1e-6, None)
        change = max(
            np.max(np.abs(gamma_new - gamma_old) / np.maximum(np.abs(gamma_old), 1e-6)),
            np.max(np.abs(delta_new - delta_old) / np.maximum(np.abs(delta_old), 1e-6)),
        )
        gamma_old, delta_old = gamma_new, delta_new
        if change < conv:
            break
    return gamma_old, delta_old


def fit_combat(frame: pd.DataFrame, batch_column: str = "cohort", features: tuple[str, ...] = FEATURES) -> ComBatContract:
    """Fit an empirical-Bayes location-and-scale batch correction (ComBat)."""

    data = frame.dropna(subset=[batch_column, *features]).copy()
    x = data[list(features)].to_numpy(float)
    grand_mean = x.mean(axis=0)
    pooled_std = x.std(axis=0, ddof=1)
    pooled_std = np.where(pooled_std > 1e-6, pooled_std, 1e-6)
    z = (x - grand_mean) / pooled_std

    batches = sorted(data[batch_column].unique())
    batch_index = {label: np.flatnonzero(data[batch_column].to_numpy() == label) for label in batches}

    gamma_hat = np.array([z[idx].mean(axis=0) for idx in batch_index.values()])
    delta_hat = np.array([z[idx].var(axis=0, ddof=1) if len(idx) > 1 else np.ones(len(features)) for idx in batch_index.values()])
    delta_hat = np.clip(delta_hat, 1e-6, None)

    gamma_bar = gamma_hat.mean(axis=1)
    tau2 = gamma_hat.var(axis=1, ddof=1)
    tau2 = np.clip(tau2, 1e-6, None)
    delta_mean = delta_hat.mean(axis=1)
    delta_var = delta_hat.var(axis=1, ddof=1)
    delta_var = np.clip(delta_var, 1e-6, None)
    a_prior = (2 * delta_var + delta_mean ** 2) / delta_var
    b_prior = (delta_mean * delta_var + delta_mean ** 3) / delta_var

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    for row, (label, idx) in enumerate(batch_index.items()):
        gamma_star[row], delta_star[row] = _it_sol(
            z[idx], gamma_hat[row], delta_hat[row], float(gamma_bar[row]), float(tau2[row]), float(a_prior[row]), float(b_prior[row])
        )

    return ComBatContract(
        features=tuple(features),
        batch_column=batch_column,
        grand_mean=pd.Series(grand_mean, index=features),
        pooled_std=pd.Series(pooled_std, index=features),
        batches=batches,
        gamma_star=pd.DataFrame(gamma_star, index=batches, columns=features),
        delta_star=pd.DataFrame(delta_star, index=batches, columns=features),
        n_per_batch=pd.Series({label: len(idx) for label, idx in batch_index.items()}),
    )


def apply_combat(frame: pd.DataFrame, contract: ComBatContract, clip: bool = True) -> pd.DataFrame:
    """Apply a fitted ComBat contract; unseen batch labels pass through unchanged."""

    output = frame.copy()
    features = list(contract.features)
    for label, indices in output.groupby(contract.batch_column).groups.items():
        if label not in contract.gamma_star.index:
            continue
        x = output.loc[indices, features].to_numpy(float)
        z = (x - contract.grand_mean.to_numpy()) / contract.pooled_std.to_numpy()
        z_adjusted = (z - contract.gamma_star.loc[label].to_numpy()) / np.sqrt(contract.delta_star.loc[label].to_numpy())
        adjusted = z_adjusted * contract.pooled_std.to_numpy() + contract.grand_mean.to_numpy()
        if clip:
            adjusted = np.clip(adjusted, 0.0, 1.0)
        output.loc[indices, features] = adjusted
    return output


def governed_combat_correction(
    frame: pd.DataFrame,
    batch_column: str = "cohort",
    features: tuple[str, ...] = FEATURES,
    *,
    min_separability_drop: float = 0.05,
    max_single_axis_shift: float = 0.35,
    seed: int = 20260822,
) -> dict:
    """Fit ComBat, apply it, and gate acceptance with an explicit, inspectable rule.

    This is the *unlabelled* sibling of the development notebook's AUROC/dose-
    response firewall (Section 11), for use where no ground-truth quality
    label exists (an ad hoc clinical batch upload). It never claims the
    correction is validated; it only reports whether two cheap, always-
    computable safety checks pass: batch separability must actually fall by a
    meaningful margin, and no batch's axis median may be dragged further than
    a plausible single step. Use the labelled cohort firewall before ever
    treating a correction as safe for pooled clinical comparison.
    """
    contract = fit_combat(frame, batch_column, features)
    corrected = apply_combat(frame, contract)

    before_test = separability_permutation_test(frame, batch_column=batch_column, features=features, n_permutations=100, seed=seed)
    separability_before = before_test["observed_separability"]
    separability_before_p_value = before_test["p_value"]
    separability_after = domain_separability(corrected, batch_column=batch_column, seed=seed)

    fingerprints_before = frame.groupby(batch_column)[list(features)].median()
    fingerprints_after = corrected.groupby(batch_column)[list(features)].median()
    max_shift = float((fingerprints_after - fingerprints_before).abs().to_numpy().max())

    separability_drop = separability_before - separability_after
    accepted = bool(separability_drop >= min_separability_drop and max_shift <= max_single_axis_shift)

    reasons = []
    if separability_drop < min_separability_drop:
        reasons.append(f"separability only fell by {separability_drop:.3f} (< {min_separability_drop:.2f} required)")
    if max_shift > max_single_axis_shift:
        reasons.append(f"largest single-axis median shift was {max_shift:.2f} (> {max_single_axis_shift:.2f} allowed)")
    if not reasons:
        reasons.append("both heuristic safety checks passed")

    return {
        "contract": contract,
        "corrected": corrected,
        "separability_before": separability_before,
        "separability_before_p_value": separability_before_p_value,
        "separability_after": separability_after,
        "separability_drop": separability_drop,
        "max_single_axis_shift": max_shift,
        "accepted": accepted,
        "reasons": reasons,
        "fingerprints_before": fingerprints_before,
        "fingerprints_after": fingerprints_after,
    }


# ---------------------------------------------------------------------------
# 3. Pathology quality-disentanglement
# ---------------------------------------------------------------------------


def pathology_quality_disentanglement(
    frame: pd.DataFrame,
    disease_column: str,
    features: tuple[str, ...] = FEATURES,
    batch_column: str | None = None,
    *,
    n_permutations: int = 500,
    seed: int = 20260822,
) -> dict:
    """Quantify how much of each quality axis tracks disease status rather than acquisition.

    For each axis, reports eta-squared (the fraction of that axis's variance
    explained by disease group) after first centering within ``batch_column``
    when given, so a disease group that happens to concentrate in one
    scanner/institution does not get credited with a quality difference that
    is really a batch effect. A high eta-squared axis is one where a FAIL/
    REVIEW route driven by that axis should be read cautiously for that
    disease group: it may be responding to pathology (haemorrhage, exudate,
    disc pallor) rather than a fixable acquisition problem.

    Each axis's eta-squared is also given a permutation-test p-value (disease
    labels are shuffled ``n_permutations`` times and eta-squared recomputed
    each time), so a small or lucky sample cannot masquerade as a genuine
    disease association: with few images per group, even a middling
    eta-squared can arise by chance, and the p-value makes that explicit
    rather than leaving the effect size to be read as if it were inferential.
    """
    data = frame.dropna(subset=[disease_column, *features]).copy()
    if batch_column is not None:
        data = data.dropna(subset=[batch_column])

    centered = data[list(features)].copy()
    if batch_column is not None:
        centered = centered - data.groupby(batch_column)[list(features)].transform("mean") + data[list(features)].mean()

    group_labels = data[disease_column].to_numpy()
    rng = np.random.default_rng(seed)

    def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
        grand_mean = values.mean()
        ss_total = float(np.sum((values - grand_mean) ** 2))
        ss_between = 0.0
        for label in np.unique(labels):
            group_values = values[labels == label]
            ss_between += len(group_values) * (group_values.mean() - grand_mean) ** 2
        return ss_between / max(ss_total, 1e-12)

    rows = []
    for feature in features:
        values = centered[feature].to_numpy(float)
        observed = _eta_squared(values, group_labels)
        if n_permutations > 0:
            null = np.empty(n_permutations)
            for i in range(n_permutations):
                null[i] = _eta_squared(values, rng.permutation(group_labels))
            p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
        else:
            p_value = float("nan")
        rows.append(
            {
                "feature": feature,
                "eta_squared": observed,
                "p_value": p_value,
                "batch_adjusted": batch_column is not None,
            }
        )
    entanglement = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)

    group_fingerprints = data.groupby(disease_column)[list(features)].median()
    group_counts = data[disease_column].value_counts()

    verdict_by_group = None
    if "verdict" in data:
        verdict_by_group = data.groupby(disease_column)["verdict"].value_counts(normalize=True).unstack(fill_value=0.0)

    return {
        "entanglement": entanglement,
        "group_fingerprints": group_fingerprints,
        "group_counts": group_counts,
        "verdict_by_group": verdict_by_group,
        "top_entangled_axis": str(entanglement.iloc[0]["feature"]) if len(entanglement) else None,
        "n_images": len(data),
        "n_groups": data[disease_column].nunique(),
    }


def separability_permutation_test(
    frame: pd.DataFrame,
    batch_column: str = "cohort",
    features: tuple[str, ...] = FEATURES,
    *,
    n_permutations: int = 200,
    seed: int = 20260822,
) -> dict:
    """Permutation test for whether batch separability exceeds what chance alone would give.

    Balanced accuracy from a quality-feature-only classifier is a point
    estimate: with a small or class-imbalanced sample, values well above 0.5
    are common by chance alone. This repeatedly shuffles the batch labels,
    recomputes :func:`eyeqc.experiments.domain_separability` on each shuffle
    to build a null distribution, and reports the fraction of shuffles at
    least as separable as the real data — an empirical, one-sided p-value
    that turns the separability number from descriptive into inferential.
    """
    observed = domain_separability(frame, batch_column=batch_column, seed=seed)
    if observed != observed:  # NaN: too few images per batch to even fit the classifier
        return {"observed_separability": observed, "p_value": float("nan"), "null_distribution": np.array([]), "n_permutations": 0}

    rng = np.random.default_rng(seed)
    labels = frame[batch_column].to_numpy()
    shuffled_frame = frame.copy()
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_frame[batch_column] = rng.permutation(labels)
        null[i] = domain_separability(shuffled_frame, batch_column=batch_column, seed=seed + i + 1)
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {"observed_separability": observed, "p_value": p_value, "null_distribution": null, "n_permutations": n_permutations}


def n_batch_domain_diagnostic(
    batches: dict[str, pd.DataFrame],
    features: tuple[str, ...] = FEATURES,
    seed: int = 20260822,
    n_permutations: int = 200,
) -> dict:
    """Generalizes the GUI's two-site diagnostic to any number of named batches."""

    frames = []
    for label, frame in batches.items():
        subset = frame[list(features)].dropna().copy()
        subset["__batch__"] = label
        frames.append(subset)
    combined = pd.concat(frames, ignore_index=True)

    separability = float("nan")
    separability_p_value = float("nan")
    counts = combined["__batch__"].value_counts()
    if len(counts) >= 2 and counts.min() >= 4:
        permutation = separability_permutation_test(
            combined, batch_column="__batch__", features=features, n_permutations=n_permutations, seed=seed
        )
        separability = permutation["observed_separability"]
        separability_p_value = permutation["p_value"]

    fingerprints = combined.groupby("__batch__")[list(features)].median()
    fingerprints_q1 = combined.groupby("__batch__")[list(features)].quantile(0.25)
    fingerprints_q3 = combined.groupby("__batch__")[list(features)].quantile(0.75)
    reference = fingerprints.mean(axis=0)
    shift = (fingerprints - reference).abs()
    dominant_axis = shift.idxmax(axis=1)
    dominant_magnitude = shift.max(axis=1)

    return {
        "n_batches": len(batches),
        "counts": counts.to_dict(),
        "domain_separability": separability,
        "domain_separability_p_value": separability_p_value,
        "fingerprints": fingerprints,
        "fingerprints_q1": fingerprints_q1,
        "fingerprints_q3": fingerprints_q3,
        "dominant_axis_per_batch": dominant_axis,
        "dominant_magnitude_per_batch": dominant_magnitude,
        "combined": combined,
    }
