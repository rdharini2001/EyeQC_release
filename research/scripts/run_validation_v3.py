"""Score every cohort with the hybrid v3 head using the operating thresholds
fixed on the development cohort alone (see train_v3_hybrid.py), applied
unchanged to every external validation cohort, and evaluate the cross-signal
agreement mechanism: does "v1's ten-axis view and v2's embedding view
disagree" actually predict where v3 is more likely to be wrong?

Thresholds are read directly from the model contract and never
recalibrated here -- this script only scores and evaluates, it does not fit
anything. (Earlier versions of this script pooled all five validation
cohorts' ground truth to recalibrate the thresholds before evaluating on
those same cohorts -- a circularity that inflated the reported NPV/triage
numbers and contradicted the manuscript's own "development-only, applied
unchanged" claim. That recalibration step has been removed.)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

CACHE = REPO_ROOT / "data_final" / "cache"
TABLES = REPO_ROOT / "data_final" / "tables"
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
        # v1's own verdict/risk_bad already ride along in `scored` from the original image_level table
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

    # ---- false-FAIL / false-PASS, v1 vs v2 vs v3
    print("\n=== false-FAIL / false-PASS: v1 vs v2 vs v3 ===")
    diag_rows = []
    pooled_counts = {k: 0 for k in ("v1_ff", "v1_fp", "v2_ff", "v2_fp", "v3_ff", "v3_fp", "good", "bad")}
    for cohort in COHORTS:
        scored = scored_cohorts[cohort]
        gt_good, gt_bad = scored["gt_bad"].eq(0), scored["gt_bad"].eq(1)
        n_good, n_bad = int(gt_good.sum()), int(gt_bad.sum())
        counts = {}
        for tag, verdict_col in (("v1", "verdict"), ("v2", "verdict_v2_only"), ("v3", "verdict_v3")):
            counts[f"{tag}_ff"] = int((gt_good & scored[verdict_col].eq("FAIL")).sum())
            counts[f"{tag}_fp"] = int((gt_bad & scored[verdict_col].eq("PASS")).sum())
        diag_rows.append(
            {
                "cohort": cohort, "N": len(scored), "gt_good": n_good, "gt_bad": n_bad,
                **{f"{k}_pct": round(100 * v / max((n_good if "ff" in k else n_bad), 1), 1) for k, v in counts.items()},
            }
        )
        for key, value in counts.items():
            pooled_counts[key] += value
        pooled_counts["good"] += n_good
        pooled_counts["bad"] += n_bad

    diag = pd.DataFrame(diag_rows)
    print(diag.to_string(index=False))
    diag.to_csv(TABLES / "v1_v2_v3_false_fail_pass.csv", index=False)

    print("\nPOOLED:")
    for tag in ("v1", "v2", "v3"):
        ff = 100 * pooled_counts[f"{tag}_ff"] / pooled_counts["good"]
        fp = 100 * pooled_counts[f"{tag}_fp"] / pooled_counts["bad"]
        print(f"  {tag}: false-FAIL={ff:.1f}%  false-PASS={fp:.1f}%")

    # ---- does v1/v2 disagreement predict v3 error? (validates the agreement signal)
    print("\n=== does v1-vs-v2 disagreement flag risky v3 calls? ===")
    pooled_scored = pd.concat(scored_cohorts.values(), ignore_index=True)
    agree = pooled_scored["verdict"] == pooled_scored["verdict_v2_only"]
    v3_wrong = (
        (pooled_scored["gt_bad"].eq(0) & pooled_scored["verdict_v3"].eq("FAIL"))
        | (pooled_scored["gt_bad"].eq(1) & pooled_scored["verdict_v3"].eq("PASS"))
    )
    print(f"  when v1 and v2 AGREE    (n={agree.sum():6d}): v3 wrong-call rate = {v3_wrong[agree].mean():.1%}")
    print(f"  when v1 and v2 DISAGREE (n={(~agree).sum():6d}): v3 wrong-call rate = {v3_wrong[~agree].mean():.1%}")

    with open(TABLES / "v3_agreement_validity.txt", "w", encoding="utf-8") as handle:
        handle.write(f"agree n={agree.sum()} wrong_rate={v3_wrong[agree].mean():.4f}\n")
        handle.write(f"disagree n={(~agree).sum()} wrong_rate={v3_wrong[~agree].mean():.4f}\n")

    print(f"\nwrote tables to {TABLES}")


if __name__ == "__main__":
    main()
