"""Deterministic EyeQC development and model freezing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .core import FEATURES


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _manifest_hash(frame: pd.DataFrame) -> str:
    columns = [column for column in ("image_id", "cluster_id", "gt_bad") if column in frame]
    payload = frame[columns].astype(str).sort_values(columns).to_csv(index=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _choose_fail_threshold(y: np.ndarray, risk: np.ndarray, target: float) -> float:
    best = None
    for threshold in np.unique(risk):
        predicted = risk >= threshold
        sensitivity = predicted[y == 1].mean()
        specificity = (~predicted[y == 0]).mean()
        if sensitivity >= target:
            candidate = (specificity, threshold)
            best = candidate if best is None or candidate > best else best
    if best is None:
        raise RuntimeError("No development threshold met the sensitivity target")
    return float(best[1])


def _choose_pass_threshold(
    y: np.ndarray,
    risk: np.ndarray,
    fail_threshold: float,
    target_npv: float,
) -> float:
    candidates = []
    for threshold in np.unique(risk[risk < fail_threshold]):
        passed = risk <= threshold
        if passed.sum() < 20:
            continue
        npv = np.mean(y[passed] == 0)
        if npv >= target_npv:
            candidates.append((passed.mean(), threshold))
    if not candidates:
        raise RuntimeError("No development threshold met the PASS NPV target")
    return float(max(candidates)[1])


def develop_eyeqc(
    development_features: pd.DataFrame,
    model_path: str | Path,
    *,
    seed: int = 20260822,
    version: str = "EyeQC-v6-CFP",
    target_bad_sensitivity: float = 0.90,
    target_pass_npv: float = 0.98,
) -> tuple[dict, pd.DataFrame]:
    """Fit, calibrate, freeze, and save EyeQC using development data only.

    Cross-validation is grouped by participant to keep paired eyes in the same
    fold. External cohorts are not accepted by this function.
    """

    required = [*FEATURES, "gt_bad", "cluster_id"]
    data = development_features.dropna(subset=required).reset_index(drop=True)
    x = data[list(FEATURES)].to_numpy(float)
    y = data["gt_bad"].astype(int).to_numpy()
    groups = data["cluster_id"].astype(str).to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = np.logspace(-2, 2, 9)
    c_scores = []

    for c_value in c_grid:
        fold_scores = []
        for train, test in cv.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            classifier = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            ).fit(scaler.transform(x[train]), y[train])
            probability = classifier.predict_proba(scaler.transform(x[test]))[:, 1]
            fold_scores.append(roc_auc_score(y[test], probability))
        c_scores.append(np.mean(fold_scores))

    best_c = float(c_grid[int(np.argmax(c_scores))])
    raw_oof = np.zeros(len(data), float)
    fold_id = np.zeros(len(data), int)

    for fold, (train, test) in enumerate(cv.split(x, y, groups), start=1):
        scaler = StandardScaler().fit(x[train])
        classifier = LogisticRegression(
            C=best_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        ).fit(scaler.transform(x[train]), y[train])
        raw_oof[test] = classifier.predict_proba(scaler.transform(x[test]))[:, 1]
        fold_id[test] = fold

    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    calibrator.fit(_logit(raw_oof).reshape(-1, 1), y)
    calibrated_oof = calibrator.predict_proba(_logit(raw_oof).reshape(-1, 1))[:, 1]

    fail_threshold = _choose_fail_threshold(y, calibrated_oof, target_bad_sensitivity)
    pass_threshold = _choose_pass_threshold(
        y,
        calibrated_oof,
        fail_threshold,
        target_pass_npv,
    )

    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        random_state=seed,
    ).fit(scaler.transform(x), y)

    feature_source = Path(__file__).with_name("core.py")
    feature_hash = hashlib.sha256(feature_source.read_bytes()).hexdigest()[:12]
    contract = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_hash": feature_hash,
        "features": list(FEATURES),
        "max_side": 640,
        "development_dataset": "EyeQ train only",
        "development_manifest_sha256": _manifest_hash(data),
        "n_development_images": int(len(data)),
        "n_development_participants": int(data["cluster_id"].nunique()),
        "cross_validation": "5-fold stratified participant-grouped",
        "regularization": "L2 logistic regression",
        "C": best_c,
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
        "logistic_intercept": float(classifier.intercept_[0]),
        "logistic_coefficients_standardized": classifier.coef_[0].tolist(),
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_coefficient": float(calibrator.coef_[0, 0]),
        "pass_risk_threshold": pass_threshold,
        "fail_risk_threshold": fail_threshold,
        "target_bad_sensitivity": target_bad_sensitivity,
        "target_pass_npv": target_pass_npv,
        "intended_use": "Technical quality triage for standard and portable colour fundus photographs",
        "excluded_use": "Ultrawide-field imaging and autonomous clinical decision making",
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    oof = data[["image_id", "cluster_id", "gt_bad"]].copy()
    oof["fold"] = fold_id
    oof["raw_probability"] = raw_oof
    oof["risk_bad"] = calibrated_oof
    oof["verdict"] = np.where(
        calibrated_oof >= fail_threshold,
        "FAIL",
        np.where(calibrated_oof <= pass_threshold, "PASS", "REVIEW"),
    )
    return contract, oof
