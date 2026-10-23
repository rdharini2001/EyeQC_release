"""Recompute failure-phenotype composition using v3's own coefficients and
current (development-only-threshold) verdicts -- the existing
contribution_* columns in image_level_v3_*.parquet belong to a different,
legacy model (model_version "EyeQC-v6-CFP-2026"), not v3's hybrid head, and
the non-PASS population composition shifted when the pooled-recalibration
threshold bug was fixed earlier this session.

Addresses a reviewer concern directly: the dominant-phenotype percentages
reported in the manuscript are computed only among REVIEW/FAIL (non-PASS)
images, never among PASS images -- this script makes that denominator
explicit in its output so the manuscript can state it plainly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.experiments import failure_phenotype_taxonomy  # noqa: E402

MODELS = REPO_ROOT / "data_final" / "models"
TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]


def main() -> None:
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    mean = np.asarray(contract["standardizer_mean"][:10])
    scale = np.asarray(contract["standardizer_scale"][:10])
    hand_coef = np.asarray(contract["logistic_coefficients_standardized"][:10])

    frames = []
    for cohort in COHORTS:
        df = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        hand = df[list(FEATURES)].to_numpy(float)
        z = (hand - mean) / scale
        # per-axis standardized contribution to the v3 logistic score (hand-feature part only)
        contrib = z * hand_coef
        for i, name in enumerate(FEATURES):
            df[f"contribution_{name}"] = contrib[:, i]
        df["verdict"] = df["verdict_v3"]  # failure_phenotype_taxonomy filters on "verdict"
        df["cohort"] = cohort
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    if "subcohort" not in pooled.columns:
        pooled["subcohort"] = pooled["cohort"]
    pooled["subcohort"] = pooled["subcohort"].fillna(pooled["cohort"])

    n_total = len(pooled)
    n_nonpass = pooled["verdict"].isin(("FAIL", "REVIEW")).sum()
    print(f"Pooled: {n_total} images, {n_nonpass} non-PASS (REVIEW/FAIL) = {100*n_nonpass/n_total:.1f}% of all images")

    # Cohort-level taxonomy (ignores subcohort) for the main-text sentence.
    subset, _ = failure_phenotype_taxonomy(pooled, verdicts=("FAIL", "REVIEW"), n_clusters=5)

    print("\n=== Cohort-level dominant-phenotype composition (denominator = non-PASS images in that cohort only) ===")
    rows = []
    for cohort in COHORTS:
        cohort_all = (pooled["cohort"] == cohort).sum()
        c = subset[subset["cohort"] == cohort]
        n_np = len(c)
        top = c["failure_phenotype"].value_counts(normalize=True).sort_values(ascending=False)
        print(f"\n{cohort}: {n_np}/{cohort_all} non-PASS ({100*n_np/cohort_all:.1f}% of cohort)")
        for phenotype, frac in top.items():
            print(f"  {phenotype}: {frac*100:.1f}% of non-PASS images")
            rows.append(
                {
                    "level": "cohort",
                    "cohort": cohort,
                    "subcohort": None,
                    "n_cohort_total": int(cohort_all),
                    "n_nonpass": int(n_np),
                    "pct_cohort_nonpass": round(100 * n_np / cohort_all, 1),
                    "failure_phenotype": phenotype,
                    "pct_of_nonpass": round(frac * 100, 1),
                }
            )

    print("\n=== Subcohort-level breakdown (MSHF-CFP and DeepDRiD, which have real subcohorts) ===")
    for cohort in ("MSHF-CFP", "DeepDRiD"):
        c = subset[subset["cohort"] == cohort]
        for sc, g in c.groupby(pooled.loc[c.index, "subcohort"]):
            top = g["failure_phenotype"].value_counts(normalize=True).sort_values(ascending=False)
            top1, frac1 = top.index[0], top.iloc[0]
            print(f"  {cohort} / {sc}: n={len(g)}  top phenotype: {top1}={frac1*100:.1f}%")
            rows.append(
                {
                    "level": "subcohort",
                    "cohort": cohort,
                    "subcohort": sc,
                    "n_cohort_total": None,
                    "n_nonpass": int(len(g)),
                    "pct_cohort_nonpass": None,
                    "failure_phenotype": top1,
                    "pct_of_nonpass": round(frac1 * 100, 1),
                }
            )

    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out_data = TABLES / "failure_phenotype_v3.csv"
    out_results = RESULTS_TABLES / "failure_phenotype_v3.csv"
    out.to_csv(out_data, index=False)
    out.to_csv(out_results, index=False)
    print(f"\nwrote {out_data}")
    print(f"wrote {out_results}")


if __name__ == "__main__":
    main()
