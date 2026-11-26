"""Consistency between separate photographs of the same eye.

Pairs are eyes with exactly two photographs.  Reported: agreement of the PASS / REVIEW / FAIL category, the share of pairs
with opposite PASS and FAIL decisions, and - where the two photographs have different reference ratings - how often
EyeQC gives the higher risk to the worse-rated photograph.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_up(frame: pd.DataFrame, eye_column: str = "eye_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    """First and second photograph (row order after sorting by index) of every eye that has exactly two."""
    counts = frame[eye_column].value_counts()
    two = frame[frame[eye_column].isin(counts[counts == 2].index)].sort_values(eye_column, kind="stable")
    first, second = two.iloc[0::2].set_index(eye_column), two.iloc[1::2].set_index(eye_column)
    return first, second


def same_eye_summary(frame: pd.DataFrame, eye_column: str = "eye_id", higher_is_worse: dict[str, bool] | None = None) -> dict:
    """``frame`` needs ``eye_column``, ``risk`` and ``verdict``; ``higher_is_worse`` names rating columns and their direction."""
    first, second = pair_up(frame, eye_column)
    if first.empty:
        raise ValueError("No eye has exactly two photographs.")
    a, b = first["verdict"].to_numpy(), second["verdict"].to_numpy()
    difference = (second["risk"] - first["risk"]).to_numpy(float)
    summary = {
        "pairs": int(len(first)),
        "same_category": float(np.mean(a == b)),
        "opposite_PASS_FAIL": float(np.mean(((a == "PASS") & (b == "FAIL")) | ((a == "FAIL") & (b == "PASS")))),
        "mean_absolute_risk_difference": float(np.mean(np.abs(difference))),
        "median_absolute_risk_difference": float(np.median(np.abs(difference))),
    }
    for column, worse_when_higher in (higher_is_worse or {}).items():
        change = (second[column] - first[column]).to_numpy(float)
        informative = np.isfinite(change) & (change != 0)
        direction = np.sign(change[informative]) * (1 if worse_when_higher else -1)
        summary[f"{column}_pairs"] = int(informative.sum())
        summary[f"{column}_higher_risk_for_worse"] = float(np.mean(np.sign(difference[informative]) == direction)) if informative.any() else float("nan")
    return summary
