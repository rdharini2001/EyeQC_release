"""How well the ten measurements predict the cohort of origin (balanced accuracy, all 21,091 photographs), before and
after location-scale normalization and ComBat harmonization.

usage: python scripts/run_cohort_separability.py
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.batch_effects import apply_combat, fit_combat  # noqa: E402
from eyeqc.core import FEATURES  # noqa: E402
from eyeqc.experiments import (  # noqa: E402
    apply_location_scale_harmonizer,
    domain_separability,
    fit_location_scale_harmonizer,
)

DATA = REPO_ROOT / "data_final"
TABLES = DATA / "tables"
SOURCE_LABELS = DATA / "source_labels"
RESULTS_TABLES = REPO_ROOT / "results" / "generated"

VALIDATION_COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
SEED = 20260822  # domain_separability's own default seed



def main() -> None:
    pooled = pd.concat(
        [pd.read_parquet(TABLES / f"image_level_v3_{c}.parquet").assign(cohort=c) for c in VALIDATION_COHORTS],
        ignore_index=True,
    )
    print(f"pooled full validation population: {len(pooled)} images across {len(VALIDATION_COHORTS)} cohorts")
    for c in VALIDATION_COHORTS:
        print(f"  {c}: {(pooled['cohort'] == c).sum()}")

    development = pd.read_parquet(SOURCE_LABELS / "development_features.parquet")
    print(f"development (reference) cohort: {len(development)} images")

    print("\nfitting ComBat (development + 5 pooled validation cohorts) and "
          "location-scale (5 pooled validation cohorts, anchored to development) ...")
    combat_source = pd.concat([development.assign(cohort="EyeQ-train"), pooled], ignore_index=True)
    combat_contract = fit_combat(combat_source, batch_column="cohort", features=FEATURES)
    ls_harmonizer = fit_location_scale_harmonizer(pooled, development, batch_column="cohort")

    ls_harmonized = apply_location_scale_harmonizer(pooled, ls_harmonizer, batch_column="cohort")
    combat_harmonized = apply_combat(pooled, combat_contract)

    print("\ncomputing domain_separability (5-fold StratifiedKFold, balanced-accuracy) on each feature set...")
    sep_raw = domain_separability(pooled, batch_column="cohort", seed=SEED)
    sep_ls = domain_separability(ls_harmonized, batch_column="cohort", seed=SEED)
    sep_combat = domain_separability(combat_harmonized, batch_column="cohort", seed=SEED)

    results = pd.DataFrame(
        [
            {"feature_set": "raw", "cohort_separability_fullscale": sep_raw},
            {"feature_set": "location_scale", "cohort_separability_fullscale": sep_ls},
            {"feature_set": "combat", "cohort_separability_fullscale": sep_combat},
        ]
    )
    results["n_images"] = len(pooled)

    print("\n=== Cohort separability (balanced accuracy), full-scale n=%d ===" % len(pooled))
    print(results.round(4).to_string(index=False))

    TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    out1 = TABLES / "cohort_separability_v3_fullscale.csv"
    out2 = RESULTS_TABLES / "cohort_separability_v3_fullscale.csv"
    results.to_csv(out1, index=False)
    results.to_csv(out2, index=False)
    print(f"\nwrote {out1}")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
