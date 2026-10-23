"""Sensitivity analysis responding to the peer-review critique that EyeQ-test
and FQS report only "resolved" binary labels, collapsing the ambiguous
"Usable" tier into "gradable" and hiding the borderline class where QC is
hardest.

The headline validation (run_validation_v3.py / Table_1_external_validation_v3.csv)
uses gt_bad_strict = (gt_label == "Reject") for the two cohorts that carry a
3-level Good/Usable/Reject scheme (EyeQ-test, FQS) -- i.e. Usable counts as
gradable, same as Good.

Here we additionally evaluate gt_bad_inclusive = (gt_label in {"Usable",
"Reject"}) -- i.e. only Good counts as gradable -- using the *same* v3 risk
scores and the *same* development-only thresholds (0.3453 PASS / 0.3861 FAIL,
read from the model contract, never touched). The model is not refit; only
the ground-truth definition used to score it changes. This isolates exactly
what the reviewer asked for: how much do AUROC / NPV / sensitivity degrade
once borderline Usable images are reclassified as poor?
"""
import json
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"
MODELS = REPO_ROOT / "data_final" / "models"

COHORTS = ["EyeQ-test", "FQS"]
BOOTSTRAPS = 1000
SEED = 20260822


def main() -> None:
    v3_contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    pass_t = float(v3_contract["pass_risk_threshold"])
    fail_t = float(v3_contract["fail_risk_threshold"])
    print(f"v3 development-only thresholds (applied unchanged): pass={pass_t:.4f} fail={fail_t:.4f}")

    rows = []
    for cohort_index, cohort in enumerate(COHORTS):
        scored = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")

        label_counts = scored["gt_label"].value_counts()
        n_total = len(scored)
        n_usable = int(label_counts.get("Usable", 0))
        n_reject = int(label_counts.get("Reject", 0))
        n_good = int(label_counts.get("Good", 0))
        print(f"\n[{cohort}] N={n_total} Good={n_good} Usable={n_usable} Reject={n_reject}")

        gt_bad_strict = (scored["gt_label"] == "Reject").astype(int)
        gt_bad_inclusive = scored["gt_label"].isin(["Usable", "Reject"]).astype(int)

        # consistency check: strict definition must equal the existing gt_bad column
        mismatch = int((gt_bad_strict.to_numpy() != scored["gt_bad"].astype(int).to_numpy()).sum())
        if mismatch:
            print(f"  WARNING: {mismatch} rows where gt_bad_strict != existing gt_bad for {cohort}")
        else:
            print(f"  consistency check OK: gt_bad_strict matches existing gt_bad column for all {n_total} rows")

        for definition, gt_bad in (("strict_reject_only", gt_bad_strict), ("inclusive_usable_plus_reject", gt_bad_inclusive)):
            eval_frame = pd.DataFrame(
                {
                    "gt_bad": gt_bad,
                    "risk_bad": scored["risk_bad_v3"],
                    "cluster_id": scored["cluster_id"],
                }
            )
            point = performance(eval_frame, pass_t, fail_t)
            intervals = cluster_bootstrap_intervals(
                eval_frame, pass_t, fail_t, bootstraps=BOOTSTRAPS, workers=4, seed=SEED + cohort_index * 10_000 + (0 if definition == "strict_reject_only" else 1)
            )
            row = {
                "cohort": cohort,
                "definition": definition,
                "N": n_total,
                "poor_N": int(gt_bad.sum()),
                "n_good": n_good,
                "n_usable": n_usable,
                "n_reject": n_reject,
                "n_usable_reclassified_to_poor": n_usable if definition == "inclusive_usable_plus_reject" else 0,
                "frac_usable_reclassified_to_poor": round(n_usable / n_total, 4) if definition == "inclusive_usable_plus_reject" else 0.0,
                **point,
            }
            for metric, (low, high) in intervals.items():
                row[f"{metric}_lo"] = low
                row[f"{metric}_hi"] = high
            rows.append(row)
            print(
                f"  [{definition}] poor_N={row['poor_N']} AUROC={point['AUROC']:.3f} "
                f"FAIL_sens={point['FAIL_sensitivity']:.3f} FAIL_spec={point['FAIL_specificity']:.3f} "
                f"PASS_NPV={point['PASS_NPV']:.3f} PASS_cov={point['PASS_coverage']:.3f}"
            )

    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out_path_1 = TABLES / "ambiguity_sensitivity_v3.csv"
    out_path_2 = RESULTS_TABLES / "ambiguity_sensitivity_v3.csv"
    out.to_csv(out_path_1, index=False)
    out.to_csv(out_path_2, index=False)
    print(f"\nwrote {out_path_1}")
    print(f"wrote {out_path_2}")
    print("\n", out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
