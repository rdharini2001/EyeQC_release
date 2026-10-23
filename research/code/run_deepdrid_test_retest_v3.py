"""Same-eye, same-session test-retest repeatability on DeepDRiD.

DeepDRiD's own acquisition protocol takes two photographs per eye per visit
(filenames end in _l1/_l2 for the left eye, _r1/_r2 for the right eye,
confirmed via file size/hash to be genuinely distinct captures, not
duplicates). The entire 1,600-image DeepDRiD cohort already used throughout
this manuscript is built from exactly these pairs, so a test-retest analysis
needs no new data -- it only needs to re-pivot the already-scored
image_level_v3_DeepDRiD.parquet table by (patient, eye).

This is same-session repeatability (two shots, one visit), not longitudinal
stability -- DeepDRiD as distributed has no multi-visit folder structure.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"

REPEAT_PATTERN = re.compile(r"(\d+)_([lr])(\d)\.jpg", re.IGNORECASE)


def parse_repeat_id(path: str) -> tuple[str, str, str] | None:
    match = REPEAT_PATTERN.search(path)
    return (match.group(1), match.group(2), match.group(3)) if match else None


def main() -> None:
    df = pd.read_parquet(TABLES / "image_level_v3_DeepDRiD.parquet")
    parsed = df["image_path"].map(parse_repeat_id)
    if parsed.isna().any():
        print(f"warning: {parsed.isna().sum()} / {len(df)} image paths did not match the l/r-repeat naming pattern; excluded")
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["patient"], df["eye"], df["rep"] = zip(*parsed)
    df["eye_id"] = df["patient"] + "_" + df["eye"]

    wide = df.pivot_table(
        index="eye_id",
        columns="rep",
        values=["risk_bad_v3", "verdict_v3", "gt_bad", "deepdrid_clarity", "deepdrid_artifact", "deepdrid_field_definition"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["risk_bad_v3_1", "risk_bad_v3_2"]).reset_index()

    sub_same = (
        (wide["deepdrid_clarity_1"] == wide["deepdrid_clarity_2"])
        & (wide["deepdrid_artifact_1"] == wide["deepdrid_artifact_2"])
        & (wide["deepdrid_field_definition_1"] == wide["deepdrid_field_definition_2"])
    )
    diff_risk = (wide["risk_bad_v3_1"] - wide["risk_bad_v3_2"]).abs()
    order = {"PASS": 2, "REVIEW": 1, "FAIL": 0}
    v1 = wide["verdict_v3_1"].map(order)
    v2 = wide["verdict_v3_2"].map(order)

    summary = pd.DataFrame(
        [
            {
                "n_eye_pairs": len(wide),
                "mean_abs_risk_diff": diff_risk.mean(),
                "median_abs_risk_diff": diff_risk.median(),
                "pearson_r_risk1_risk2": pearsonr(wide["risk_bad_v3_1"], wide["risk_bad_v3_2"])[0],
                "verdict_exact_agreement": (wide["verdict_v3_1"] == wide["verdict_v3_2"]).mean(),
                "verdict_full_flip_PASS_FAIL": ((v1 - v2).abs() == 2).mean(),
                "gt_bad_consistent_between_repeats": (wide["gt_bad_1"] == wide["gt_bad_2"]).mean(),
                "n_pairs_identical_dataset_subscores": int(sub_same.sum()),
                "mean_abs_risk_diff_when_subscores_identical": diff_risk[sub_same].mean(),
                "mean_abs_risk_diff_when_subscores_differ": diff_risk[~sub_same].mean(),
            }
        ]
    )
    print(summary.T)

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "deepdrid_test_retest_summary_v3.csv", index=False)
    summary.to_csv(RESULTS_TABLES / "deepdrid_test_retest_summary_v3.csv", index=False)
    wide.to_csv(TABLES / "deepdrid_test_retest_pairs_v3.csv", index=False)
    wide.to_csv(RESULTS_TABLES / "deepdrid_test_retest_pairs_v3.csv", index=False)
    print(f"\nwrote {TABLES / 'deepdrid_test_retest_summary_v3.csv'}")
    print(f"wrote {RESULTS_TABLES / 'deepdrid_test_retest_summary_v3.csv'}")


if __name__ == "__main__":
    main()
