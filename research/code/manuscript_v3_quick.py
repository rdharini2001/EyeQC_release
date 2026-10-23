"""Recompute every manuscript number that only needs the already-cached v3
risk scores (no new image processing / no new embeddings) — construct
validity, subdomain AUROC, baseline comparison, decision-curve analysis,
local recalibration, and harmonization safety (ComBat features already
saved; location-scale harmonized on the fly with the same vendored fitter
v1 used). All of this reuses generic, unmodified functions from the
vendored eyeqc/ package — only the risk score being fed in changes."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.experiments import expected_calibration_error, robust_location_scale_harmonize  # noqa: E402

DATA = REPO_ROOT / "data_final"
TABLES = DATA / "tables"
V1_TABLES = Path(r"F:\EyeQC\EyeQC_npj\archive\paper_outputs_v10_end_to_end\tables")
OUT = TABLES  # v3 manuscript numbers land alongside the other v3 tables
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
SEED = 20260822


def load_v3(cohort: str) -> pd.DataFrame:
    return pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")


def all_v3() -> pd.DataFrame:
    return pd.concat([load_v3(c) for c in COHORTS], ignore_index=True)


# ---------------------------------------------------------------- Panel B: construct validity
def construct_validity() -> None:
    checks = [
        ("FQS", "mos", -1),
        ("MSHF-CFP", "gt_clarity", -1),
        ("MSHF-CFP", "gt_contrast", -1),
        ("MSHF-CFP", "gt_illumination", -1),
        ("DeepDRiD", "deepdrid_artifact", 1),
        ("DeepDRiD", "deepdrid_clarity", -1),
        ("DeepDRiD", "deepdrid_field_definition", -1),
    ]
    rows = []
    for cohort, column, direction in checks:
        data = load_v3(cohort).dropna(subset=[column, "risk_bad_v3"])
        rho, p = spearmanr(data["risk_bad_v3"], data[column])
        rows.append({"cohort": cohort, "reference_measure": column, "N": len(data), "Spearman_rho": rho, "expected_direction": direction, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "construct_validity_v3.csv", index=False)
    print("\n=== construct_validity_v3 ===")
    print(out.round(4).to_string(index=False))


# ---------------------------------------------------------------- Supp 1B / bonus: subdomain + stratum AUROC
def subdomain_auroc() -> None:
    rows = []
    for cohort, column in [("MSHF-CFP", "subcohort"), ("DeepDRiD", "source")]:
        data = load_v3(cohort)
        for subdomain, group in data.groupby(column):
            if group["gt_bad"].nunique() < 2:
                auroc = np.nan
            else:
                auroc = roc_auc_score(group["gt_bad"], group["risk_bad_v3"])
            rows.append(
                {
                    "cohort": cohort,
                    "subdomain": subdomain,
                    "N": len(group),
                    "poor_fraction": group["gt_bad"].mean(),
                    "median_risk": group["risk_bad_v3"].median(),
                    "AUROC": auroc,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "institution_scanner_subdomain_validation_v3.csv", index=False)
    print("\n=== institution_scanner_subdomain_validation_v3 ===")
    print(out.round(4).to_string(index=False))

    dr = load_v3("DeepDRiD")
    if "dr_grade" in dr.columns:
        grade_col = "dr_grade"
    else:
        grade_col = None
    if grade_col:
        rows = []
        for grade, group in dr.groupby(grade_col):
            auroc = roc_auc_score(group["gt_bad"], group["risk_bad_v3"]) if group["gt_bad"].nunique() == 2 else np.nan
            rows.append({"dr_grade": grade, "N": len(group), "AUROC": auroc})
        pd.DataFrame(rows).to_csv(OUT / "deepdrid_stratum_auroc_v3.csv", index=False)
        print("\n=== deepdrid_stratum_auroc_v3 ===")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    else:
        print("\n(no dr_grade column in v3 DeepDRiD table — skipping stratum AUROC)")


# ---------------------------------------------------------------- Supp 2B: baseline comparison
def baseline_comparison() -> None:
    v1_baselines = pd.read_csv(V1_TABLES / "baseline_heuristic_comparison.csv")
    v3_auroc = pd.read_csv(TABLES / "Table_1_external_validation_v3.csv")[["cohort", "AUROC"]].rename(
        columns={"AUROC": "EyeQC v3 (hybrid)"}
    )
    merged = v1_baselines.drop(columns=["EyeQC (ten interpretable axes)"]).merge(v3_auroc, on="cohort")
    merged = merged.rename(columns={"EyeQC v3 (hybrid)": "EyeQC v3 (hybrid, ten axes + embedding)"})
    merged.to_csv(OUT / "baseline_heuristic_comparison_v3.csv", index=False)
    print("\n=== baseline_heuristic_comparison_v3 ===")
    print(merged.round(4).to_string(index=False))


# ---------------------------------------------------------------- Supp 8A: decision curve
def decision_curve() -> None:
    thresholds = np.arange(0.05, 0.6501, 0.0125)
    rows = []
    for cohort in COHORTS:
        data = load_v3(cohort).dropna(subset=["gt_bad", "risk_bad_v3"])
        y = data["gt_bad"].to_numpy(int)
        risk = data["risk_bad_v3"].to_numpy(float)
        n = len(y)
        prevalence = y.mean()
        for t in thresholds:
            positive = risk >= t
            tp = np.sum(positive & (y == 1))
            fp = np.sum(positive & (y == 0))
            nb_eyeqc = tp / n - fp / n * (t / (1 - t))
            nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
            nb_none = 0.0
            rows.append(
                {"threshold": round(t, 4), "net_benefit_eyeqc": nb_eyeqc, "net_benefit_reacquire_all": nb_all,
                 "net_benefit_reacquire_none": nb_none, "cohort": cohort}
            )
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "decision_curve_net_benefit_v3.csv", index=False)

    summary_rows = []
    for cohort, g in curve.groupby("cohort"):
        delta_all = g["net_benefit_eyeqc"] - g["net_benefit_reacquire_all"]
        delta_none = g["net_benefit_eyeqc"] - g["net_benefit_reacquire_none"]
        summary_rows.append(
            {
                "cohort": cohort, "thresholds": len(g),
                "EyeQC_gt_None": int((delta_none > 0).sum()),
                "EyeQC_gt_All": int((delta_all > 0).sum()),
                "min_dNB_vs_All": delta_all.min(), "max_dNB_vs_All": delta_all.max(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "decision_curve_policy_summary_v3.csv", index=False)
    print("\n=== decision_curve_policy_summary_v3 ===")
    print(summary.round(4).to_string(index=False))


# ---------------------------------------------------------------- Supp 7B: local recalibration
def local_recalibration() -> None:
    from sklearn.linear_model import LogisticRegression

    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    rows = []
    rng = np.random.default_rng(SEED)
    for cohort in COHORTS:
        data = load_v3(cohort).dropna(subset=["gt_bad", "risk_bad_v3"]).reset_index(drop=True)
        y_all = data["gt_bad"].to_numpy(int)
        risk_all = data["risk_bad_v3"].to_numpy(float)
        ece_before = expected_calibration_error(y_all, risk_all)
        brier_before = brier_score_loss(y_all, risk_all)
        for n_cal in (25, 50, 100, 200):
            if n_cal >= len(data):
                continue
            idx = rng.choice(len(data), size=n_cal, replace=False)
            mask = np.zeros(len(data), bool)
            mask[idx] = True
            if len(set(y_all[mask])) < 2:
                continue
            calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            calibrator.fit(_logit(risk_all[mask]).reshape(-1, 1), y_all[mask])
            held_out = ~mask
            recalibrated = calibrator.predict_proba(_logit(risk_all[held_out]).reshape(-1, 1))[:, 1]
            ece_after = expected_calibration_error(y_all[held_out], recalibrated)
            brier_after = brier_score_loss(y_all[held_out], recalibrated)
            rows.append(
                {
                    "cohort": cohort, "sample_size": n_cal,
                    "median_ECE_before": ece_before, "median_ECE_after": ece_after,
                    "median_Brier_before": brier_before, "median_Brier_after": brier_after,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "local_recalibration_summary_v3.csv", index=False)
    print("\n=== local_recalibration_summary_v3 ===")
    print(out.round(4).to_string(index=False))


# ---------------------------------------------------------------- Supp 6: harmonization safety with v3
def harmonization_safety() -> None:
    harmonized_combat = pd.read_parquet(V1_TABLES / "combat_harmonized_features_all_external_cohorts.parquet")

    contract_path = DATA / "models" / "eyeqc_v3_model.json"
    import json

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])

    def score(hand: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        x = np.concatenate([hand, embedding], axis=1)
        z = (x - mean) / scale
        raw_logit = intercept + z @ coef
        calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
        return 1 / (1 + np.exp(-calibrated_logit))

    combat_rows, ls_rows = [], []
    for cohort in COHORTS:
        v3_data = load_v3(cohort).reset_index(drop=True)
        embeddings = np.load(REPO_ROOT / "data_final" / "cache" / f"{cohort}_dinov2_embeddings.npy")
        ok = np.load(REPO_ROOT / "data_final" / "cache" / f"{cohort}_dinov2_ok.npy")
        index = pd.read_parquet(REPO_ROOT / "data_final" / "cache" / f"{cohort}_index.parquet")
        index, embeddings = index[ok].reset_index(drop=True), embeddings[ok]

        merged = index[["image_id", "gt_bad"] + list(FEATURES)].merge(
            harmonized_combat[["image_id"] + list(FEATURES)], on="image_id", suffixes=("", "_harm")
        )
        emb_map = {iid: embeddings[i] for i, iid in enumerate(index["image_id"])}
        merged = merged[merged["image_id"].isin(emb_map)].reset_index(drop=True)
        emb_aligned = np.stack([emb_map[iid] for iid in merged["image_id"]])

        raw_hand = merged[list(FEATURES)].to_numpy(float)
        harm_hand = merged[[f"{f}_harm" for f in FEATURES]].to_numpy(float)

        risk_before = score(raw_hand, emb_aligned)
        risk_after_combat = score(harm_hand, emb_aligned)
        auroc_before = roc_auc_score(merged["gt_bad"], risk_before)
        auroc_after_combat = roc_auc_score(merged["gt_bad"], risk_after_combat)
        combat_rows.append(
            {"cohort": cohort, "AUROC_before": auroc_before, "AUROC_after": auroc_after_combat,
             "AUROC_change": auroc_after_combat - auroc_before}
        )

    # location-scale: reference = EyeQ-train (development set), exactly as v1's notebook used it
    all_raw = pd.concat([load_v3(c)[list(FEATURES) + ["cohort"]] for c in COHORTS], ignore_index=True)
    development = pd.read_parquet(DATA / "source_labels" / "development_features.parquet")
    ls_harmonized = robust_location_scale_harmonize(all_raw, development, batch_column="cohort")

    for cohort in COHORTS:
        v3_data = load_v3(cohort).reset_index(drop=True)
        embeddings = np.load(REPO_ROOT / "data_final" / "cache" / f"{cohort}_dinov2_embeddings.npy")
        ok = np.load(REPO_ROOT / "data_final" / "cache" / f"{cohort}_dinov2_ok.npy")
        embeddings = embeddings[ok]
        mask = all_raw["cohort"] == cohort
        harm_hand = ls_harmonized.loc[mask, list(FEATURES)].to_numpy(float)
        n = min(len(harm_hand), len(embeddings))
        risk_after_ls = score(harm_hand[:n], embeddings[:n])
        gt = v3_data["gt_bad"].to_numpy()[:n]
        auroc_before = roc_auc_score(v3_data["gt_bad"].to_numpy()[:n], v3_data["risk_bad_v3"].to_numpy()[:n])
        auroc_after_ls = roc_auc_score(gt, risk_after_ls)
        ls_rows.append(
            {"cohort": cohort, "AUROC_before": auroc_before, "AUROC_after": auroc_after_ls,
             "AUROC_change": auroc_after_ls - auroc_before}
        )

    pd.DataFrame(combat_rows).to_csv(OUT / "harmonization_safety_combat_v3.csv", index=False)
    pd.DataFrame(ls_rows).to_csv(OUT / "harmonization_safety_location_scale_v3.csv", index=False)
    print("\n=== harmonization_safety_combat_v3 ===")
    print(pd.DataFrame(combat_rows).round(4).to_string(index=False))
    print("\n=== harmonization_safety_location_scale_v3 ===")
    print(pd.DataFrame(ls_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    construct_validity()
    subdomain_auroc()
    baseline_comparison()
    decision_curve()
    local_recalibration()
    harmonization_safety()
    print("\nAll quick v3 manuscript numbers written to", OUT)
