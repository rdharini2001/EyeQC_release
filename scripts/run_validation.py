"""Score every cohort with the released model and its fixed development thresholds (nothing is fitted here).

Writes image-level predictions and the cohort performance table with participant-bootstrap confidence intervals.
usage: python scripts/run_validation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

CACHE = REPO_ROOT / "data_final" / "cache"
TABLES = REPO_ROOT / "data_final" / "tables"
MODELS = REPO_ROOT / "models"

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
BOOTSTRAPS = 1000
SEED = 20260822


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def score(x: np.ndarray, contract: dict) -> np.ndarray:
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])
    z = (x - mean) / scale
    raw_logit = intercept + z @ coef
    calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
    return _sigmoid(calibrated_logit)


def main() -> None:
    v3_contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    v2_contract = json.loads((MODELS / "eyeqc_v2_model.json").read_text(encoding="utf-8"))

    scored_cohorts = {}
    for cohort in COHORTS:
        index = pd.read_parquet(CACHE / f"{cohort}_index.parquet")
        embeddings = np.load(CACHE / f"{cohort}_dinov2_embeddings.npy")
        ok = np.load(CACHE / f"{cohort}_dinov2_ok.npy")
        ok = ok & index[list(FEATURES)].notna().all(axis=1).to_numpy()
        index, embeddings = index[ok].reset_index(drop=True), embeddings[ok]

        x_hybrid = np.concatenate([index[list(FEATURES)].to_numpy(float), embeddings], axis=1)
        risk_v3 = score(x_hybrid, v3_contract)
        risk_v2 = score(embeddings, v2_contract)

        out = index.copy()
        out["risk_bad_v3"] = risk_v3
        out["risk_bad_v2"] = risk_v2
        scored_cohorts[cohort] = out

    # ---- thresholds come from the development-cohort fit, applied unchanged
    pass_t = float(v3_contract["pass_risk_threshold"])
    fail_t = float(v3_contract["fail_risk_threshold"])
    print(f"v3 development-only thresholds (applied unchanged): pass={pass_t:.4f} fail={fail_t:.4f}")

    v2_pass_t = float(v2_contract["pass_risk_threshold"])
    v2_fail_t = float(v2_contract["fail_risk_threshold"])

    def verdict_of(risk: pd.Series, pass_t: float, fail_t: float) -> pd.Series:
        return pd.Series(np.where(risk >= fail_t, "FAIL", np.where(risk <= pass_t, "PASS", "REVIEW")), index=risk.index)

    performance_rows = []
    for cohort_index, cohort in enumerate(COHORTS):
        scored = scored_cohorts[cohort]
        scored["verdict_v3"] = verdict_of(scored["risk_bad_v3"], pass_t, fail_t)
        scored["verdict_v2_only"] = verdict_of(scored["risk_bad_v2"], v2_pass_t, v2_fail_t)
        scored_cohorts[cohort] = scored
        scored.to_parquet(TABLES / f"image_level_v3_{cohort}.parquet", index=False)

        eval_frame = pd.DataFrame({"gt_bad": scored["gt_bad"], "risk_bad": scored["risk_bad_v3"], "cluster_id": scored["cluster_id"]})
        point = performance(eval_frame, pass_t, fail_t)
        intervals = cluster_bootstrap_intervals(eval_frame, pass_t, fail_t, bootstraps=BOOTSTRAPS, workers=4, seed=SEED + cohort_index * 10_000)
        row = {"cohort": cohort, "N": len(scored), "poor_N": int(scored["gt_bad"].sum()), **point}
        for metric, (low, high) in intervals.items():
            row[f"{metric}_lo"] = low
            row[f"{metric}_hi"] = high
        performance_rows.append(row)
        print(f"[{cohort}] N={len(scored)} AUROC={point['AUROC']:.3f} FAIL_sens={point['FAIL_sensitivity']:.3f} PASS_NPV={point['PASS_NPV']:.3f}")

    table1 = pd.DataFrame(performance_rows)
    table1.to_csv(TABLES / "Table_1_external_validation_v3.csv", index=False)
    print("\n", table1.round(3).to_string(index=False))

    print(f"\nwrote tables to {TABLES}")


if __name__ == "__main__":
    main()
