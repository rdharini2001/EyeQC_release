"""EyeQC v3: a hybrid decision head that fuses v1's ten dedicated
focus/exposure/contrast heuristics with the frozen DINOv2 embedding into a
single calibrated logistic head — the model actually shipped.

Why: the DINOv2-only variant fixed false-FAIL/false-PASS agreement with
each cohort's own ground truth but got *worse* at tracking synthetic
degradation severity for defocus/overexposure/haze — DINOv2's self-
supervised pretraining uses blur/exposure augmentation specifically to make
its features invariant to those changes, which is backwards for a quality
detector. v1's hand-crafted features have the opposite profile: built to be
sensitive to exactly those three things, but calibrated in a way that
doesn't transfer cleanly across acquisition conditions (the original
diagnosed problem). Concatenating both into one decision gives the joint
model access to both kinds of evidence at once — see results_final/REPORT.md
section 7 for the full before/after numbers.

A separate idea — flagging disagreement between v1's and the DINOv2-only
model's verdicts as a confidence signal — was tested in run_validation_v3.py
and *rejected*: disagreement did not actually predict where the hybrid model
was more likely to be wrong. It is not part of this model.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold, _logit  # noqa: E402

CACHE = REPO_ROOT / "data_final" / "cache"
MODELS = REPO_ROOT / "data_final" / "models"
TABLES = REPO_ROOT / "data_final" / "tables"
MODELS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

SEED = 20260822
VERSION = "EyeQC-v3-hybrid-CFP-2026"
TARGET_BAD_SENSITIVITY = 0.90
TARGET_PASS_NPV = 0.98
C_GRID = np.logspace(-4, 2, 13)


def build_design_matrix(index: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    hand_features = index[list(FEATURES)].to_numpy(float)
    return np.concatenate([hand_features, embeddings], axis=1)


def fit_logistic_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, label: str) -> dict:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    c_scores = []
    for c_value in C_GRID:
        fold_scores = []
        for train, test in cv.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            clf = LogisticRegression(C=c_value, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
            clf.fit(scaler.transform(x[train]), y[train])
            fold_scores.append(roc_auc_score(y[test], clf.predict_proba(scaler.transform(x[test]))[:, 1]))
        c_scores.append(float(np.mean(fold_scores)))
    best_c = float(C_GRID[int(np.argmax(c_scores))])
    print(f"[{label}] selected C={best_c:.5g} (CV AUROC={max(c_scores):.4f})")

    raw_oof = np.zeros(len(x), float)
    for train, test in cv.split(x, y, groups):
        scaler = StandardScaler().fit(x[train])
        clf = LogisticRegression(C=best_c, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
        clf.fit(scaler.transform(x[train]), y[train])
        raw_oof[test] = clf.predict_proba(scaler.transform(x[test]))[:, 1]

    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    calibrator.fit(_logit(raw_oof).reshape(-1, 1), y)
    calibrated_oof = calibrator.predict_proba(_logit(raw_oof).reshape(-1, 1))[:, 1]
    oof_auroc = roc_auc_score(y, calibrated_oof)
    print(f"[{label}] out-of-fold AUROC = {oof_auroc:.4f}")

    fail_t = _choose_fail_threshold(y, calibrated_oof, TARGET_BAD_SENSITIVITY)
    pass_t = _choose_pass_threshold(y, calibrated_oof, fail_t, TARGET_PASS_NPV)

    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=best_c, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=SEED)
    clf.fit(scaler.transform(x), y)

    return {
        "best_c": best_c,
        "oof_auroc": oof_auroc,
        "calibrated_oof": calibrated_oof,
        "fail_threshold": fail_t,
        "pass_threshold": pass_t,
        "standardizer_mean": scaler.mean_.tolist(),
        "standardizer_scale": scaler.scale_.tolist(),
        "logistic_intercept": float(clf.intercept_[0]),
        "logistic_coefficients_standardized": clf.coef_[0].tolist(),
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_coefficient": float(calibrator.coef_[0, 0]),
    }


def main() -> None:
    index = pd.read_parquet(CACHE / "development_index.parquet")
    embeddings = np.load(CACHE / "development_dinov2_embeddings.npy")
    ok = np.load(CACHE / "development_dinov2_ok.npy")
    ok = ok & index[list(FEATURES)].notna().all(axis=1).to_numpy()

    data = index[ok].reset_index(drop=True)
    embeddings = embeddings[ok]
    y = data["gt_bad"].astype(int).to_numpy()
    groups = data["cluster_id"].astype(str).to_numpy()
    print(f"development set: {len(data)} images, {len(set(groups))} participants, {y.mean():.1%} bad")

    x_hybrid = build_design_matrix(data, embeddings)
    print(f"hybrid design matrix: {x_hybrid.shape} (10 hand features + {embeddings.shape[1]}-dim embedding)")

    fit = fit_logistic_cv(x_hybrid, y, groups, "hybrid")

    manifest_payload = data[["image_id", "cluster_id", "gt_bad"]].astype(str).sort_values(
        ["image_id", "cluster_id", "gt_bad"]
    ).to_csv(index=False)
    manifest_hash = hashlib.sha256(manifest_payload.encode()).hexdigest()

    contract = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "hybrid: 10 hand-crafted quality axes concatenated with a frozen dinov2_vitb14 embedding, "
        "single calibrated logistic head",
        "hand_features": list(FEATURES),
        "backbone": "dinov2_vitb14",
        "embedding_dim": int(embeddings.shape[1]),
        "input_dim": int(x_hybrid.shape[1]),
        "development_dataset": "EyeQ train only",
        "development_manifest_sha256": manifest_hash,
        "n_development_images": int(len(data)),
        "n_development_participants": int(len(set(groups))),
        "cross_validation": "5-fold stratified participant-grouped",
        "regularization": "L2 logistic regression over [hand features || DINOv2 embedding]",
        "C": fit["best_c"],
        "standardizer_mean": fit["standardizer_mean"],
        "standardizer_scale": fit["standardizer_scale"],
        "logistic_intercept": fit["logistic_intercept"],
        "logistic_coefficients_standardized": fit["logistic_coefficients_standardized"],
        "platt_intercept": fit["platt_intercept"],
        "platt_coefficient": fit["platt_coefficient"],
        "pass_risk_threshold": fit["pass_threshold"],
        "fail_risk_threshold": fit["fail_threshold"],
        "threshold_calibration": "Fit on the development cohort alone (five-fold out-of-fold predictions); "
        "applied unchanged to all five external validation cohorts.",
        "target_bad_sensitivity": TARGET_BAD_SENSITIVITY,
        "target_pass_npv": TARGET_PASS_NPV,
        "development_oof_auroc": fit["oof_auroc"],
        "intended_use": "Technical quality triage for standard and portable colour fundus photographs",
        "excluded_use": "Ultrawide-field imaging and autonomous clinical decision making",
        "predecessor": "EyeQC-v2-robust-CFP-2026 (DINOv2 embedding only); superseded because it lost sensitivity "
        "to synthetic defocus/overexposure/haze severity (median risk change ~0), see REPORT.md section 6.",
    }
    (MODELS / "eyeqc_v3_model.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"wrote {MODELS / 'eyeqc_v3_model.json'}")

    oof = data[["image_id", "cluster_id", "gt_bad"]].copy()
    oof["risk_bad"] = fit["calibrated_oof"]
    oof["verdict"] = np.where(
        fit["calibrated_oof"] >= fit["fail_threshold"], "FAIL",
        np.where(fit["calibrated_oof"] <= fit["pass_threshold"], "PASS", "REVIEW"),
    )
    oof.to_parquet(TABLES / "development_oof_predictions_v3.parquet", index=False)


if __name__ == "__main__":
    main()
