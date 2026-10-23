"""Mechanistic, transport, and reader-standardization experiments."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import FEATURES, EyeQCModel, extract_features
from .pipeline import read_rgb


DEGRADATIONS = (
    "defocus",
    "darkness",
    "overexposure",
    "haze",
    "jpeg",
    "glare",
    "local_blur",
    "shadow",
    "decentring",
)


def apply_degradation(
    rgb: np.ndarray,
    kind: str,
    dose: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply one deterministic acquisition-like degradation at dose 0–1."""

    image = rgb.astype(np.float32)
    height, width = image.shape[:2]
    mask = None

    if dose == 0:
        return rgb.copy(), None

    if kind == "defocus":
        sigma = 0.5 + 7.5 * dose
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    elif kind == "darkness":
        image *= 1 - 0.80 * dose
    elif kind == "overexposure":
        image = image * (1 - 0.65 * dose) + 255 * 0.65 * dose
    elif kind == "haze":
        image = image * (1 - 0.60 * dose) + 175 * 0.60 * dose
    elif kind == "jpeg":
        quality = int(round(95 - 90 * dose))
        bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif kind in {"glare", "local_blur", "shadow"}:
        center = (int(rng.uniform(0.30, 0.70) * width), int(rng.uniform(0.30, 0.70) * height))
        axes = (int((0.08 + 0.13 * dose) * width), int((0.08 + 0.13 * dose) * height))
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
        alpha = mask[..., None].astype(float)
        if kind == "glare":
            image = image * (1 - alpha * 0.90 * dose) + 255 * alpha * 0.90 * dose
        elif kind == "shadow":
            image *= 1 - alpha * 0.85 * dose
        else:
            blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1 + 10 * dose)
            image = image * (1 - alpha) + blurred * alpha
        mask = mask.astype(bool)
    elif kind == "decentring":
        shift_x = int(width * 0.32 * dose)
        shift_y = int(height * 0.18 * dose)
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        image = cv2.warpAffine(image, matrix, (width, height), borderValue=(0, 0, 0))
    else:
        raise ValueError(f"Unknown degradation: {kind}")

    return np.clip(image, 0, 255).astype(np.uint8), mask


def tile_quality_map(features: dict) -> np.ndarray:
    grid = np.full((6, 6), np.nan)
    for tile in features["tiles"]:
        grid[int(tile["yi"]), int(tile["xi"])] = tile["quality"]
    fill = np.nanmedian(grid) if np.isfinite(grid).any() else 0.0
    return np.nan_to_num(grid, nan=fill)


def _localization_metrics(grid: np.ndarray, mask: np.ndarray) -> dict:
    resized_mask = cv2.resize(mask.astype(np.uint8), (6, 6), interpolation=cv2.INTER_AREA) >= 0.25
    predicted = grid <= np.quantile(grid, 0.20)
    intersection = np.logical_and(predicted, resized_mask).sum()
    union = np.logical_or(predicted, resized_mask).sum()
    minimum = np.unravel_index(np.argmin(grid), grid.shape)
    return {
        "localization_iou": intersection / union if union else np.nan,
        "localization_point_hit": float(resized_mask[minimum]),
        "localization_contrast": float(grid[~resized_mask].mean() - grid[resized_mask].mean()),
    }


def _degrade_one_image(
    image_path: str,
    image_id: str,
    cohort: str,
    model_path: str,
    doses: tuple[float, ...],
    seed: int,
) -> list[dict]:
    model = EyeQCModel(model_path)
    rgb = read_rgb(image_path)
    baseline_features = extract_features(rgb)
    baseline_score = model.score_features(baseline_features)
    rows = []

    for kind_index, kind in enumerate(DEGRADATIONS):
        for dose in doses:
            rng = np.random.default_rng(seed + kind_index * 10_000 + int(dose * 1000))
            if dose == 0:
                features = baseline_features
                score = baseline_score
                mask = None
            else:
                degraded, mask = apply_degradation(rgb, kind, dose, rng)
                features = extract_features(degraded)
                score = model.score_features(features)
            row = {
                "image_id": image_id,
                "cohort": cohort,
                "degradation": kind,
                "dose": dose,
                "risk_bad": score["risk_bad"],
                "verdict": score["verdict"],
            }
            row.update({axis: features[axis] for axis in FEATURES})
            if mask is not None:
                row.update(_localization_metrics(tile_quality_map(features), mask))
            rows.append(row)
    return rows


def run_degradation_study(
    sample: pd.DataFrame,
    model_path: str | Path,
    output_path: str | Path,
    *,
    doses: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
    workers: int = 4,
    seed: int = 20260822,
) -> pd.DataFrame:
    """Run large-scale dose response and spatial localization with one task per image."""

    output_path = Path(output_path)
    if output_path.exists():
        return pd.read_parquet(output_path)

    tasks = Parallel(n_jobs=workers, prefer="processes")(
        delayed(_degrade_one_image)(
            row.image_path,
            str(row.image_id),
            str(row.cohort),
            str(model_path),
            doses,
            seed + index * 100_000,
        )
        for index, row in sample.reset_index(drop=True).iterrows()
    )
    output = pd.DataFrame([record for image_rows in tasks for record in image_rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output


def degradation_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, image_id), group in results.sort_values("dose").groupby(["degradation", "image_id"]):
        risk = group["risk_bad"].to_numpy()
        rows.append(
            {
                "degradation": kind,
                "image_id": image_id,
                "monotonic": bool(np.all(np.diff(risk) >= -1e-6)),
                "risk_change": float(risk[-1] - risk[0]),
            }
        )
    image_level = pd.DataFrame(rows)
    return (
        image_level.groupby("degradation")
        .agg(images=("image_id", "size"), monotonic_fraction=("monotonic", "mean"), median_risk_change=("risk_change", "median"))
        .reset_index()
    )


def _compound_one_image(
    image_path: str,
    image_id: str,
    model_path: str,
    doses: tuple[float, ...],
    seed: int,
) -> list[dict]:
    model = EyeQCModel(model_path)
    rgb = read_rgb(image_path)

    def risk(image: np.ndarray) -> float:
        return model.score_features(extract_features(image))["risk_bad"]

    baseline = risk(rgb)
    blur_images = {}
    blur_risks = {}
    dark_risks = {}
    for dose in doses:
        blur_images[dose], _ = apply_degradation(rgb, "defocus", dose, np.random.default_rng(seed))
        blur_risks[dose] = risk(blur_images[dose])
        dark, _ = apply_degradation(rgb, "darkness", dose, np.random.default_rng(seed + 1))
        dark_risks[dose] = risk(dark)
    rows = []
    for blur_dose in doses:
        blurred = blur_images[blur_dose]
        blur_risk = blur_risks[blur_dose]
        for dark_dose in doses:
            dark_risk = dark_risks[dark_dose]
            combined, _ = apply_degradation(blurred, "darkness", dark_dose, np.random.default_rng(seed + 2))
            combined_risk = risk(combined)
            logit = lambda value: np.log(np.clip(value, 1e-6, 1 - 1e-6) / np.clip(1 - value, 1e-6, 1))
            interaction = logit(combined_risk) - (logit(blur_risk) + logit(dark_risk) - logit(baseline))
            rows.append(
                {
                    "image_id": image_id,
                    "blur_dose": blur_dose,
                    "dark_dose": dark_dose,
                    "baseline_risk": baseline,
                    "blur_risk": blur_risk,
                    "dark_risk": dark_risk,
                    "combined_risk": combined_risk,
                    "logit_interaction": float(interaction),
                }
            )
    return rows


def run_compound_study(
    sample: pd.DataFrame,
    model_path: str | Path,
    output_path: str | Path,
    *,
    doses: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    workers: int = 4,
    seed: int = 20260822,
) -> pd.DataFrame:
    """Quantify non-additive response to simultaneous blur and underexposure."""

    output_path = Path(output_path)
    if output_path.exists():
        return pd.read_parquet(output_path)
    tasks = Parallel(n_jobs=workers, prefer="processes")(
        delayed(_compound_one_image)(row.image_path, str(row.image_id), str(model_path), doses, seed + index)
        for index, row in sample.reset_index(drop=True).iterrows()
    )
    output = pd.DataFrame([record for image_rows in tasks for record in image_rows])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return output


def fit_location_scale_harmonizer(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    batch_column: str = "cohort",
) -> pd.DataFrame:
    """Fit robust per-batch location and scale against a reference."""

    reference_median = reference[list(FEATURES)].median()
    reference_iqr = (reference[list(FEATURES)].quantile(0.75) - reference[list(FEATURES)].quantile(0.25)).clip(lower=1e-3)
    rows = []
    for batch_name, batch in frame.groupby(batch_column):
        batch_median = batch[list(FEATURES)].median()
        batch_iqr = (batch[list(FEATURES)].quantile(0.75) - batch[list(FEATURES)].quantile(0.25)).clip(lower=1e-3)
        for feature in FEATURES:
            rows.append(
                {
                    batch_column: batch_name,
                    "feature": feature,
                    "batch_median": batch_median[feature],
                    "batch_iqr": batch_iqr[feature],
                    "reference_median": reference_median[feature],
                    "reference_iqr": reference_iqr[feature],
                }
            )
    return pd.DataFrame(rows)


def apply_location_scale_harmonizer(
    frame: pd.DataFrame,
    parameters: pd.DataFrame,
    batch_column: str = "cohort",
) -> pd.DataFrame:
    """Apply previously fitted robust location-scale parameters."""

    output = frame.copy()
    for batch_name, indices in output.groupby(batch_column).groups.items():
        batch_parameters = parameters[parameters[batch_column].eq(batch_name)].set_index("feature")
        if len(batch_parameters) != len(FEATURES):
            raise ValueError(f"No complete harmonization contract for batch {batch_name}")
        for feature in FEATURES:
            row = batch_parameters.loc[feature]
            output.loc[indices, feature] = (
                (output.loc[indices, feature] - row["batch_median"])
                / row["batch_iqr"]
                * row["reference_iqr"]
                + row["reference_median"]
            ).clip(0, 1)
    return output


def robust_location_scale_harmonize(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    batch_column: str = "cohort",
) -> pd.DataFrame:
    """Exploratory feature harmonization aligned to a development reference.

    This transformation is deliberately separate from validated scoring. It
    must pass discrimination and dose-response retention checks before use.
    """

    parameters = fit_location_scale_harmonizer(frame, reference, batch_column)
    return apply_location_scale_harmonizer(frame, parameters, batch_column)


def domain_separability(
    frame: pd.DataFrame,
    batch_column: str = "cohort",
    seed: int = 20260822,
    n_splits: int = 5,
) -> float:
    """Balanced accuracy of a quality-feature-only batch classifier.

    A quality-quantitative firewall metric: if EyeQC's ten axes alone can
    identify which scanner or institution an image came from, the batches
    differ systematically and any pooled comparison of quality between them
    is confounded. ``n_splits`` self-clamps to the smallest batch so ad hoc
    two-batch GUI comparisons do not need a separate code path.
    """

    labels = frame[batch_column].astype("category").cat.codes.to_numpy()
    n_splits = max(2, min(n_splits, np.bincount(labels).min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    predicted = cross_val_predict(model, frame[list(FEATURES)], labels, cv=splitter)
    return float(balanced_accuracy_score(labels, predicted))


def failure_phenotype_taxonomy(
    scored: pd.DataFrame,
    *,
    verdicts: tuple[str, ...] = ("FAIL", "REVIEW"),
    n_clusters: int = 5,
    seed: int = 20260822,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster non-PASS images by axis-contribution profile into recurring failure phenotypes.

    Clusters are labelled by their dominant (most negative) median contributing
    axis so a phenotype corresponds to an actionable acquisition failure mode
    (for example "focus-dominant" or "illumination_uniformity-dominant") rather
    than an opaque cluster index. This is a scanner/institution-agnostic failure
    taxonomy; :func:`batch_effect_fingerprint` reports where each phenotype
    concentrates across cohorts and scanners.
    """

    contribution_columns = [f"contribution_{axis}" for axis in FEATURES]
    subset = scored[scored["verdict"].isin(verdicts)].dropna(subset=contribution_columns).copy()
    if len(subset) < n_clusters:
        raise ValueError("Too few non-PASS cases for the requested number of failure phenotypes")

    x = StandardScaler().fit_transform(subset[contribution_columns])
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x)
    subset["phenotype_cluster"] = labels

    cluster_axis = subset.groupby("phenotype_cluster")[contribution_columns].median()
    dominant_axis = cluster_axis.idxmin(axis=1).str.replace("contribution_", "", regex=False)
    subset["failure_phenotype"] = subset["phenotype_cluster"].map(dominant_axis)

    group_columns = [column for column in ("cohort", "subcohort") if column in subset]
    composition = (
        subset.groupby(group_columns)["failure_phenotype"]
        .value_counts(normalize=True)
        .rename("fraction")
        .reset_index()
    )
    return subset, composition


def batch_effect_fingerprint(harmonizer: pd.DataFrame, batch_column: str = "cohort") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank the feature-level shift responsible for each batch's harmonization correction.

    Reuses the per-batch/per-feature median and IQR already fitted by
    :func:`fit_location_scale_harmonizer`. Ranking uses the raw median
    difference in native [0, 1] axis-score units rather than an IQR-standardized
    effect size: several axes (for example focus and color balance) are nearly
    saturated in the development reference, so their reference IQR is close to
    zero and a standardized ratio explodes without indicating a real problem.
    The standardized column is kept for context with a floored denominator.
    """

    ranked = harmonizer.copy()
    ranked["raw_shift"] = ranked["batch_median"] - ranked["reference_median"]
    ranked["standardized_shift"] = ranked["raw_shift"] / ranked["reference_iqr"].clip(lower=0.05)
    ranked = ranked.sort_values("raw_shift", key=np.abs, ascending=False)
    dominant = (
        ranked.groupby(batch_column)
        .first()[["feature", "raw_shift", "standardized_shift"]]
        .rename(columns={"feature": "dominant_shift_axis", "raw_shift": "dominant_shift_magnitude", "standardized_shift": "dominant_shift_standardized"})
        .reset_index()
    )
    return ranked, dominant


def two_site_domain_diagnostic(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label_a: str = "Site A",
    label_b: str = "Site B",
    seed: int = 20260822,
) -> dict:
    """Lightweight two-batch harmonization diagnostic for ad hoc GUI comparisons.

    Mirrors the notebook's cohort-scale domain-separability and fingerprint-shift
    diagnostics at the scale of a live clinician-facing comparison of two
    acquisition batches (for example two scanners, two sites, or before/after a
    protocol change). It answers the same two questions the harmonization
    firewall answers for whole cohorts: are these batches distinguishable on
    quality features alone, and if so, which axis is driving it.
    """

    columns = list(FEATURES)
    a = frame_a[columns].dropna().copy()
    b = frame_b[columns].dropna().copy()
    combined = pd.concat([a.assign(__site__=label_a), b.assign(__site__=label_b)], ignore_index=True)

    separability = float("nan")
    if min(len(a), len(b)) >= 4:
        separability = domain_separability(combined, batch_column="__site__", seed=seed)

    fingerprint_a = a.median()
    fingerprint_b = b.median()
    # Raw median difference in native [0, 1] axis-score units: robust to axes
    # that are nearly saturated (near-zero IQR) in one or both batches, unlike
    # an IQR-standardized effect size, which would explode on those axes.
    shift = (fingerprint_a - fingerprint_b).sort_values(key=np.abs, ascending=False)

    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "domain_separability": separability,
        "fingerprint_a": fingerprint_a,
        "fingerprint_b": fingerprint_b,
        "axis_shift": shift,
        "dominant_shift_axis": str(shift.index[0]) if len(shift) else None,
        "dominant_shift_magnitude": float(shift.iloc[0]) if len(shift) else float("nan"),
    }


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    groups = np.digitize(probability, edges[1:-1])
    error = 0.0
    for group in range(bins):
        selected = groups == group
        if selected.any():
            error += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(error)


def _intercept_recalibration(y: np.ndarray, probability: np.ndarray) -> float:
    logits = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))

    def loss(intercept: float) -> float:
        calibrated = 1 / (1 + np.exp(-(logits + intercept)))
        return brier_score_loss(y, calibrated)

    return float(minimize_scalar(loss, bounds=(-6, 6), method="bounded").x)


def simulate_local_recalibration(
    frame: pd.DataFrame,
    sample_sizes: tuple[int, ...] = (25, 50, 100, 200),
    repeats: int = 100,
    seed: int = 20260822,
) -> pd.DataFrame:
    """Estimate the value of small locally labelled calibration samples."""

    rng = np.random.default_rng(seed)
    rows = []
    y_all = frame["gt_bad"].astype(int).to_numpy()
    p_all = frame["risk_bad"].to_numpy(float)
    for sample_size in sample_sizes:
        if sample_size >= len(frame) - 20:
            continue
        for repeat in range(repeats):
            calibration = rng.choice(len(frame), size=sample_size, replace=False)
            evaluation = np.setdiff1d(np.arange(len(frame)), calibration)
            intercept = _intercept_recalibration(y_all[calibration], p_all[calibration])
            logits = np.log(np.clip(p_all[evaluation], 1e-6, 1 - 1e-6) / np.clip(1 - p_all[evaluation], 1e-6, 1))
            calibrated = 1 / (1 + np.exp(-(logits + intercept)))
            rows.append(
                {
                    "sample_size": sample_size,
                    "repeat": repeat,
                    "ece_before": expected_calibration_error(y_all[evaluation], p_all[evaluation]),
                    "ece_after": expected_calibration_error(y_all[evaluation], calibrated),
                    "brier_before": brier_score_loss(y_all[evaluation], p_all[evaluation]),
                    "brier_after": brier_score_loss(y_all[evaluation], calibrated),
                    "auroc_after": roc_auc_score(y_all[evaluation], calibrated),
                }
            )
    return pd.DataFrame(rows)


def reader_operating_thresholds(reader_cases: pd.DataFrame) -> pd.DataFrame:
    """Find reader-specific EyeQC cut-points that best reproduce ordinal grades."""

    rows = []
    candidates = np.quantile(reader_cases["risk_bad"], np.linspace(0.02, 0.98, 60))
    for reader in sorted(reader_cases["reader"].unique()):
        data = reader_cases[reader_cases["reader"].eq(reader)].dropna(subset=["class_ordinal"])
        best = None
        for pass_threshold in candidates:
            for fail_threshold in candidates[candidates > pass_threshold]:
                predicted = np.where(data["risk_bad"] >= fail_threshold, 0, np.where(data["risk_bad"] <= pass_threshold, 2, 1))
                loss = np.mean(np.abs(predicted - data["class_ordinal"]))
                candidate = (loss, pass_threshold, fail_threshold)
                best = candidate if best is None or candidate < best else best
        rows.append({"reader": reader, "mean_absolute_category_error": best[0], "pass_risk_threshold": best[1], "fail_risk_threshold": best[2]})
    return pd.DataFrame(rows)
