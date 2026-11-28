"""Spearman correlation between EyeQC poor-quality risk and each cohort's independent image-quality ratings.

Signs follow the direction of the original scale.  Needs the scored cohort tables under data_final/tables/.
usage: python scripts/run_rating_correlations.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "data_final" / "tables"
OUTPUT = REPO_ROOT / "results" / "generated"
RATINGS = {"FQS": ("mos",), "MSHF-CFP": ("gt_clarity", "gt_contrast", "gt_illumination"),
           "DeepDRiD": ("deepdrid_artifact", "deepdrid_clarity", "deepdrid_field_definition")}


def main() -> None:
    rows = []
    for cohort, columns in RATINGS.items():
        table = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        for column in columns:
            data = table.dropna(subset=[column, "risk_bad_v3"])
            result = spearmanr(data.risk_bad_v3, data[column])
            rows.append({"cohort": cohort, "reference_measure": column, "N": len(data), "Spearman_rho": result.statistic, "p_value": result.pvalue})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "rating_correlations.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
