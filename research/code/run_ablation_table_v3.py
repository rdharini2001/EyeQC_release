"""Ablation table: v1 (handcrafted-only) vs v2 (DINOv2-only) vs v3 (hybrid),
scored on all five external validation cohorts, using each model's OWN fixed,
already-determined operating thresholds read straight from its model
contract. No recalibration happens here -- this script only scores and
evaluates, exactly like run_validation_v3.py, whose scoring pattern it
reuses for v2/v3 and extends to v1.

Model contracts:
  v1 (handcrafted-only, 10-axis logistic regression, no embedding)
      F:\\EyeQC\\EyeQC_npj\\EyeQC_end_to_end\\models\\eyeqc_model.json
      (version "EyeQC-v6-CFP-2026" -- confirmed as v2's own recorded
      "predecessor" in data_final/models/eyeqc_v2_model.json)
  v2 (DINOv2-only, embedding logistic regression, no handcrafted features)
      data_final/models/eyeqc_v2_model.json
  v3 (hybrid: 10 handcrafted axes concatenated with the DINOv2 embedding)
      data_final/models/eyeqc_v3_model.json

All three are scored on the identical cached per-cohort index/embedding
files in data_final/cache/ (no new feature extraction), and on the
identical image subset (the same dinov2-embedding-availability + hand
feature completeness filter run_validation_v3.py uses) so the three model
arms are compared on an identical population per cohort.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
EYEQC_NPJ_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

CACHE = REPO_ROOT / "data_final" / "cache"
TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"
MODELS = REPO_ROOT / "data_final" / "models"
V1_MODEL_PATH = EYEQC_NPJ_ROOT / "EyeQC_end_to_end" / "models" / "eyeqc_model.json"

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


def verdict_of(risk: pd.Series, pass_t: float, fail_t: float) -> pd.Series:
    return pd.Series(
        np.where(risk >= fail_t, "FAIL", np.where(risk <= pass_t, "PASS", "REVIEW")),
        index=risk.index,
    )


def main() -> None:
    v1_contract = json.loads(V1_MODEL_PATH.read_text(encoding="utf-8"))
    v2_contract = json.loads((MODELS / "eyeqc_v2_model.json").read_text(encoding="utf-8"))
    v3_contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))

    assert tuple(v1_contract["features"]) == tuple(FEATURES), "v1 feature order mismatch"
    assert v2_contract.get("predecessor", "").startswith(v1_contract["version"]), (
        "v2 contract's recorded predecessor does not match the v1 contract found -- "
        "double-check this is really v1."
    )

    print(f"v1 contract : {V1_MODEL_PATH}  version={v1_contract['version']}")
    print(f"v2 contract : {MODELS / 'eyeqc_v2_model.json'}  version={v2_contract['version']}")
    print(f"v3 contract : {MODELS / 'eyeqc_v3_model.json'}  version={v3_contract['version']}")

    models = {
        "v1_handcrafted_only": (v1_contract, "hand"),
        "v2_dinov2_only": (v2_contract, "embed"),
        "v3_hybrid": (v3_contract, "hybrid"),
    }
    for tag, (contract, _) in models.items():
        print(
            f"  {tag}: pass_risk_threshold={contract['pass_risk_threshold']:.6f} "
            f"fail_risk_threshold={contract['fail_risk_threshold']:.6f}"
        )

    rows = []
    for cohort_index, cohort in enumerate(COHORTS):
        index = pd.read_parquet(CACHE / f"{cohort}_index.parquet")
        embeddings = np.load(CACHE / f"{cohort}_dinov2_embeddings.npy")
        ok = np.load(CACHE / f"{cohort}_dinov2_ok.npy")
        ok = ok & index[list(FEATURES)].notna().all(axis=1).to_numpy()
        index, embeddings = index[ok].reset_index(drop=True), embeddings[ok]

        x_hand = index[list(FEATURES)].to_numpy(float)
        x_hybrid = np.concatenate([x_hand, embeddings], axis=1)
        x_of = {"hand": x_hand, "embed": embeddings, "hybrid": x_hybrid}

        for tag, (contract, kind) in models.items():
            risk = score(x_of[kind], contract)
            pass_t = float(contract["pass_risk_threshold"])
            fail_t = float(contract["fail_risk_threshold"])
            eval_frame = pd.DataFrame(
                {"gt_bad": index["gt_bad"], "risk_bad": risk, "cluster_id": index["cluster_id"]}
            )
            point = performance(eval_frame, pass_t, fail_t)
            intervals = cluster_bootstrap_intervals(
                eval_frame, pass_t, fail_t, bootstraps=BOOTSTRAPS, workers=4,
                seed=SEED + cohort_index * 10_000 + hash(tag) % 1000,
            )
            row = {
                "cohort": cohort, "model": tag, "N": len(index), "poor_N": int(index["gt_bad"].sum()),
                "pass_risk_threshold": pass_t, "fail_risk_threshold": fail_t, **point,
            }
            for metric, (low, high) in intervals.items():
                row[f"{metric}_lo"] = low
                row[f"{metric}_hi"] = high
            rows.append(row)
            print(
                f"[{cohort}] {tag}: N={len(index)} AUROC={point['AUROC']:.3f} "
                f"AUPRC={point['AUPRC']:.3f} Brier={point['Brier']:.3f} "
                f"FAIL_sens={point['FAIL_sensitivity']:.3f} PASS_NPV={point['PASS_NPV']:.3f}"
            )

    table = pd.DataFrame(rows)
    cohort_order = {c: i for i, c in enumerate(COHORTS)}
    model_order = {m: i for i, m in enumerate(models)}
    table = table.sort_values(
        by=["cohort", "model"], key=lambda s: s.map(cohort_order) if s.name == "cohort" else s.map(model_order)
    ).reset_index(drop=True)

    out_main = TABLES / "ablation_v1_v2_v3_full_v3.csv"
    out_copy = RESULTS_TABLES / "ablation_v1_v2_v3_full_v3.csv"
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_main, index=False)
    table.to_csv(out_copy, index=False)

    print("\n=== Ablation table (v1 handcrafted-only vs v2 DINOv2-only vs v3 hybrid) ===")
    display_cols = ["cohort", "model", "N", "poor_N", "AUROC", "AUPRC", "Brier", "FAIL_sensitivity", "PASS_NPV"]
    print(table[display_cols].round(4).to_string(index=False))
    print(f"\nwrote {out_main}")
    print(f"wrote {out_copy}")


if __name__ == "__main__":
    main()
