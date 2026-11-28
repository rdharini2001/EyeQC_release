"""Consistency between separate photographs of the same eye (DeepDRiD).

DeepDRiD file names encode participant, eye and acquisition (``<participant>_<l|r><number>.jpg``).  Eyes with exactly two
photographs form the pairs (798 in the study); eyes with another number of photographs are excluded.  For each rating
(artifacts, clarity, field definition), pairs with different ratings are scored as concordant when the worse-rated
photograph has the higher risk (ties count half); 95% intervals come from 1,000 participant resamples.

usage: python scripts/run_same_eye_pairs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.pairs import pair_up, same_eye_summary  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
OUTPUT = REPO_ROOT / "results" / "generated"
PATTERN = r"(\d+)_([lr])(\d+)\.jpg"
RATINGS = {"deepdrid_artifact": True, "deepdrid_clarity": False, "deepdrid_field_definition": False}   # True: higher is worse
BOOTSTRAPS, SEED = 1000, 20261024 + 200


def participant_weights(participants, rng: np.random.Generator) -> np.ndarray:
    """How often each row appears when participants are resampled with replacement."""
    _, index = np.unique(np.asarray(participants).astype(str), return_inverse=True)
    count = index.max() + 1
    return np.bincount(rng.integers(0, count, count), minlength=count)[index]


def main() -> None:
    table = pd.read_parquet(TABLES / "image_level_v3_DeepDRiD.parquet")
    parts = table.image_path.str.replace("\\", "/", regex=False).str.extract(PATTERN, flags=2)
    keep = parts[0].notna()
    frame = pd.DataFrame({"participant": parts[0], "eye_id": parts[0] + "_" + parts[1].str.lower(), "acquisition": pd.to_numeric(parts[2]),
                          "risk": table.risk_bad_v3, "verdict": table.verdict_v3, **{name: table[name] for name in RATINGS}})[keep]
    frame = frame.sort_values(["eye_id", "acquisition"]).reset_index(drop=True)
    summary = same_eye_summary(frame, higher_is_worse=RATINGS)
    first, second = pair_up(frame)

    rows = [{"quantity": key, "value": value} for key, value in summary.items()]
    intervals = {}
    for name, worse_when_higher in RATINGS.items():
        change = ((second[name] - first[name]) * (1 if worse_when_higher else -1)).to_numpy(float)
        informative = np.isfinite(change) & (change != 0)
        agreement = np.sign((second.risk - first.risk).to_numpy(float)[informative]) * np.sign(change[informative])
        score = np.where(agreement > 0, 1.0, np.where(agreement == 0, 0.5, 0.0))
        participants = first.participant.to_numpy()[informative]
        rng = np.random.default_rng(SEED)
        draws = [np.average(score, weights=participant_weights(participants, rng)) for _ in range(BOOTSTRAPS)]
        intervals[f"{name}_higher_risk_for_worse"] = np.quantile(draws, [0.025, 0.975])
    for row in rows:
        if row["quantity"] in intervals:
            row["ci_low"], row["ci_high"] = intervals[row["quantity"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows, columns=["quantity", "value", "ci_low", "ci_high"])
    result.to_csv(OUTPUT / "same_eye_pairs.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
