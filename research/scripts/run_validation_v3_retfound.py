"""Score all 5 external cohorts with the RETFound-backbone hybrid model
(``eyeqc_v3_retfound_model.json``) using its own development-only-fixed
thresholds, applied UNCHANGED -- no recalibration on external data. This
mirrors the exact anti-circularity protocol already fixed in
``run_validation_v3.py`` for the real (DINOv2) v3 model: thresholds are read
directly from the model contract and never refit here; this script only
scores and evaluates.

Ground truth (``gt_bad``) and cluster ids are taken unmodified from each
cohort's existing ``{cohort}_index.parquet`` -- never redefined here.

Output: one row per cohort (cohort, N, poor_N, AUROC, AUPRC, Brier,
FAIL_sensitivity, PASS_NPV), dual-saved to data_final/tables and
results_final/tables as baseline_retfound_v3.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

CACHE = REPO_ROOT / "data_final" / "cache"
TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"
MODELS = REPO_ROOT / "data_final" / "models"

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
    contract = json.loads((MODELS / "eyeqc_v3_retfound_model.json").read_text(encoding="utf-8"))

    pass_t = float(contract["pass_risk_threshold"])
    fail_t = float(contract["fail_risk_threshold"])
    print(f"RETFound-v3 development-only thresholds (applied unchanged): pass={pass_t:.4f} fail={fail_t:.4f}")

    rows = []
    failed_counts = {}
    for cohort_index, cohort in enumerate(COHORTS):
        index = pd.read_parquet(CACHE / f"{cohort}_index.parquet")
        embeddings = np.load(CACHE / f"{cohort}_retfound_embeddings.npy")
        embed_ok = np.load(CACHE / f"{cohort}_retfound_ok.npy")
        feature_ok = index[list(FEATURES)].notna().all(axis=1).to_numpy()
        ok = embed_ok & feature_ok
        n_total = len(index)
        n_embed_failed = int((~embed_ok).sum())
        n_feature_failed = int((~feature_ok & embed_ok).sum())
        failed_counts[cohort] = {
            "n_total": n_total,
            "n_embedding_failed": n_embed_failed,
            "n_feature_nan_only": n_feature_failed,
            "n_used": int(ok.sum()),
        }

        scored = index[ok].reset_index(drop=True)
        embeddings = embeddings[ok]
        x_hybrid = np.concatenate([scored[list(FEATURES)].to_numpy(float), embeddings], axis=1)
        scored["risk_bad"] = score(x_hybrid, contract)
        scored.to_parquet(TABLES / f"image_level_v3_retfound_{cohort}.parquet", index=False)

        eval_frame = pd.DataFrame(
            {"gt_bad": scored["gt_bad"], "risk_bad": scored["risk_bad"], "cluster_id": scored["cluster_id"]}
        )
        point = performance(eval_frame, pass_t, fail_t)
        intervals = cluster_bootstrap_intervals(
            eval_frame, pass_t, fail_t, bootstraps=BOOTSTRAPS, workers=4, seed=SEED + cohort_index * 10_000
        )
        row = {
            "cohort": cohort,
            "N": len(scored),
            "poor_N": int(scored["gt_bad"].sum()),
            "AUROC": point["AUROC"],
            "AUPRC": point["AUPRC"],
            "Brier": point["Brier"],
            "FAIL_sensitivity": point["FAIL_sensitivity"],
            "PASS_NPV": point["PASS_NPV"],
        }
        rows.append(row)
        print(
            f"[{cohort}] N={len(scored)} (of {n_total}, {n_embed_failed} embedding failures) "
            f"AUROC={point['AUROC']:.3f} AUPRC={point['AUPRC']:.3f} Brier={point['Brier']:.3f} "
            f"FAIL_sens={point['FAIL_sensitivity']:.3f} PASS_NPV={point['PASS_NPV']:.3f} "
            f"[AUROC 95% CI {intervals['AUROC'][0]:.3f}-{intervals['AUROC'][1]:.3f}]"
        )

    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "baseline_retfound_v3.csv", index=False)
    table.to_csv(RESULTS_TABLES / "baseline_retfound_v3.csv", index=False)
    print(f"\nwrote baseline_retfound_v3.csv to {TABLES} and {RESULTS_TABLES}")
    print("\n", table.round(4).to_string(index=False))

    print("\nembedding-extraction failure counts per cohort:")
    for cohort, counts in failed_counts.items():
        print(f"  {cohort}: {counts}")


if __name__ == "__main__":
    main()
