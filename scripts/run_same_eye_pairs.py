"""Consistency between separate photographs of the same eye (DeepDRiD).

DeepDRiD file names encode participant, eye and acquisition (``<participant>_<l|r><number>.jpg``).  Eyes with exactly two
photographs form the pairs (798 in the study); eyes with another number of photographs are excluded.  95% intervals come
from resampling participants.

usage: python scripts/run_same_eye_pairs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.pairs import same_eye_summary  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
OUTPUT = REPO_ROOT / "results" / "generated"
PATTERN = r"(\d+)_([lr])(\d+)\.jpg"
RATINGS = {"deepdrid_artifact": True, "deepdrid_clarity": False, "deepdrid_field_definition": False}   # True: higher is worse
BOOTSTRAPS, SEED = 2000, 20260822


def main() -> None:
    table = pd.read_parquet(TABLES / "image_level_v3_DeepDRiD.parquet")
    parts = table.image_path.str.replace("\\", "/", regex=False).str.extract(PATTERN, flags=2)
    keep = parts[0].notna()
    frame = pd.DataFrame({"participant": parts[0], "eye_id": parts[0] + "_" + parts[1].str.lower(), "acquisition": pd.to_numeric(parts[2]),
                          "risk": table.risk_bad_v3, "verdict": table.verdict_v3, **{name: table[name] for name in RATINGS}})[keep]
    frame = frame.sort_values(["eye_id", "acquisition"]).reset_index(drop=True)
    summary = same_eye_summary(frame, higher_is_worse=RATINGS)

    rng = np.random.default_rng(SEED)
    members = [group for _, group in frame.groupby("participant")]
    draws = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(members), len(members))
        sample = pd.concat([members[i].assign(eye_id=members[i].eye_id + f"#{k}") for k, i in enumerate(chosen)], ignore_index=True)
        draws.append(same_eye_summary(sample, higher_is_worse=RATINGS))
    draws = pd.DataFrame(draws)
    rows = [{"quantity": key, "value": value,
             **({} if key.endswith("pairs") else {"ci_low": draws[key].quantile(0.025), "ci_high": draws[key].quantile(0.975)})}
            for key, value in summary.items()]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "same_eye_pairs.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
