"""Association of the ten measurements with diabetic retinopathy grade in DeepDRiD (eta squared after centring within
sub-cohort, permutation p values).

usage: python scripts/run_disease_association.py
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.batch_effects import pathology_quality_disentanglement  # noqa: E402
from eyeqc.core import FEATURES, DISPLAY_NAMES  # noqa: E402

DATA = REPO_ROOT / "data_final"
TABLES = DATA / "tables"
RESULTS_TABLES = REPO_ROOT / "results" / "generated"



def main() -> None:
    ddr = pd.read_parquet(TABLES / "image_level_v3_DeepDRiD.parquet")
    print(f"image_level_v3_DeepDRiD.parquet: {len(ddr)} rows (full DeepDRiD cohort)")
    n_missing_dr = ddr["dr_level"].isna().sum()
    print(f"dr_level missing: {n_missing_dr}/{len(ddr)}")
    print("dr_level distribution:\n", ddr["dr_level"].value_counts(dropna=False).sort_index())
    print("subcohort distribution:\n", ddr["subcohort"].value_counts(dropna=False))

    result = pathology_quality_disentanglement(
        ddr, disease_column="dr_level", features=FEATURES, batch_column="subcohort",
    )
    entanglement = result["entanglement"].copy()
    entanglement["display_name"] = entanglement["feature"].map(DISPLAY_NAMES)
    entanglement["n_images_used"] = result["n_images"]
    entanglement["n_dr_groups"] = result["n_groups"]

    print(f"\nn_images used = {result['n_images']} (full cohort = {len(ddr)})")
    print(f"n DR severity groups = {result['n_groups']}")
    print("\n=== Disease (DR)-quality disentanglement eta^2, full 1,600-image DeepDRiD cohort, batch-adjusted (subcohort) ===")
    print(entanglement.round(4).to_string(index=False))

    top = entanglement.iloc[0]
    print(
        "CONFIRMED: this analysis already used the full 1,600-image DeepDRiD cohort "
        "(no larger or smaller DeepDRiD quality table exists in data_final/)."
        if len(ddr) == 1600 and n_missing_dr == 0
        else "WARNING: cohort is not full/complete -- see counts above."
    )

    entanglement["is_full_cohort_check"] = (len(ddr) == 1600 and n_missing_dr == 0)

    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out1 = TABLES / "disease_quality_disentanglement_v3_fullcohort_check.csv"
    out2 = RESULTS_TABLES / "disease_quality_disentanglement_v3_fullcohort_check.csv"
    entanglement.to_csv(out1, index=False)
    entanglement.to_csv(out2, index=False)
    print(f"\nwrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
