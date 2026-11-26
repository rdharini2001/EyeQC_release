"""Local probability recalibration across cohorts and label budgets (participant-separated repeated splits).

For each cohort, whole participants are drawn into a labelled set of at most the budget, the monotone logistic
correction of ``eyeqc.recalibration`` is fitted there, and frozen and corrected risks are compared on everyone else.

usage: python scripts/run_local_recalibration.py [--repeats 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.recalibration import fit_recalibration, labelled_participants, operating_metrics  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
OUTPUT = REPO_ROOT / "results" / "generated"
COHORTS = ("EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD")
BUDGETS = (25, 50, 100, 200)
SEED = 20261024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    rows = []
    for cohort_index, cohort in enumerate(COHORTS):
        table = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        labels, risk = table.gt_bad.to_numpy(int), table.risk_bad_v3.to_numpy()
        groups = table.cluster_id.astype(str).to_numpy()
        for budget in BUDGETS:
            if budget >= len(table):
                continue
            for repeat in range(args.repeats):
                rng = np.random.default_rng(SEED + 10000 + cohort_index * 100000 + budget * 100 + repeat)
                fit_rows = labelled_participants(groups, budget, rng)
                if np.unique(labels[fit_rows]).size < 2:
                    continue
                mapping = fit_recalibration(risk[fit_rows], labels[fit_rows])
                held = ~fit_rows
                for method, values in (("frozen", risk[held]), ("recalibrated", mapping(risk[held]))):
                    rows.append(dict(cohort=cohort, budget=budget, repeat=repeat, method=method,
                                     labelled=int(fit_rows.sum()), **operating_metrics(labels[held], values)))
            print(cohort, budget, "done", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    summary = raw.drop(columns="repeat").groupby(["cohort", "budget", "method"], as_index=False).median()
    raw.to_csv(OUTPUT / "local_recalibration_repeats.csv", index=False)
    summary.to_csv(OUTPUT / "local_recalibration_summary.csv", index=False)
    print(summary[summary.budget == 100][["cohort", "method", "ECE", "Brier", "false_pass_rate", "pass_share"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
