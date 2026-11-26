"""
Task 1: Calibration / reliability diagrams per external cohort for EyeQC v3.

Builds binned reliability-diagram data (predicted risk vs observed frequency of
gt_bad=1) for each of the 5 external cohorts, using the already-computed
risk_bad / gt_bad columns in image_level_v3_{cohort}.parquet.

Binning scheme: 10 equal-WIDTH bins on predicted risk in [0,1] by default, but
falls back to merging any bin with n < MIN_BIN_N (20) into its neighbor so that
observed-frequency estimates are not driven by a handful of points. This keeps
bin edges interpretable ("risk 0.0-0.1", etc.) rather than cohort-specific
equal-count edges, at the cost of some sparse bins in the tails for small
cohorts (DRIMDB n=194, MSHF-CFP n=802, DeepDRiD n=1600) -- which is exactly the
sparsity the merge step handles.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

BASE = REPOSITORY
DATA_DIR = BASE / "data_final" / "tables"
OUT_DATA = BASE / "data_final" / "tables" / "calibration_reliability_v3.csv"
OUT_TABLE_COPY = BASE / "results" / "generated" / "calibration_reliability_v3.csv"

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
N_BINS = 10
MIN_BIN_N = 20


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(bin_df, n_total):
    # weighted mean abs gap, weighted by bin count
    w = bin_df["n"] / n_total
    gap = (bin_df["mean_predicted_risk"] - bin_df["observed_frequency"]).abs()
    return float((w * gap).sum())


def build_bins(risk, gt, n_bins=N_BINS, min_bin_n=MIN_BIN_N):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(risk, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        rows.append({
            "bin": b,
            "bin_lo": edges[b],
            "bin_hi": edges[b + 1],
            "n": n,
            "mean_predicted_risk": float(risk[mask].mean()) if n > 0 else np.nan,
            "observed_frequency": float(gt[mask].mean()) if n > 0 else np.nan,
            "n_positive": int(gt[mask].sum()) if n > 0 else 0,
        })
    df = pd.DataFrame(rows)

    # Merge sparse bins (n < min_bin_n) into the nearest non-empty neighbor,
    # preferring to merge toward the adjacent bin with larger n, iteratively,
    # until every retained bin has n >= min_bin_n or only one bin remains.
    df["merged_into"] = df["bin"]
    changed = True
    while changed:
        changed = False
        active = df[df["merged_into"] == df["bin"]]
        # recompute n per active group
        grp = df.groupby("merged_into").agg(n=("n", "sum")).reset_index()
        sparse = grp[(grp["n"] > 0) & (grp["n"] < min_bin_n)]
        if sparse.empty or len(grp) <= 1:
            break
        # take the lowest-index sparse group, merge into its neighbor group
        # with the smaller current bin_hi gap (prefer merging downward/left
        # unless it's bin 0, then merge upward/right)
        target_bin = int(sparse.iloc[0]["merged_into"])
        groups_sorted = sorted(grp["merged_into"].tolist())
        pos = groups_sorted.index(target_bin)
        if pos == 0:
            neighbor = groups_sorted[1] if len(groups_sorted) > 1 else target_bin
        elif pos == len(groups_sorted) - 1:
            neighbor = groups_sorted[pos - 1]
        else:
            # merge with whichever neighbor currently has fewer points (grow the smaller side less skew) -
            # simplest: merge with the left neighbor
            neighbor = groups_sorted[pos - 1]
        if neighbor == target_bin:
            break
        df.loc[df["merged_into"] == target_bin, "merged_into"] = neighbor
        changed = True

    out_rows = []
    for g, sub in df.groupby("merged_into"):
        n = int(sub["n"].sum())
        if n == 0:
            continue
        npos = int(sub["n_positive"].sum())
        lo = float(sub["bin_lo"].min())
        hi = float(sub["bin_hi"].max())
        mean_pred = float((sub["mean_predicted_risk"] * sub["n"]).sum() / n)
        out_rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "n": n,
            "n_positive": npos,
            "mean_predicted_risk": mean_pred,
            "observed_frequency": npos / n,
        })
    out = pd.DataFrame(out_rows).sort_values("bin_lo").reset_index(drop=True)
    out.insert(0, "bin", range(len(out)))
    return out


def main():
    all_rows = []
    summary_rows = []
    for cohort in COHORTS:
        path = DATA_DIR / f"image_level_v3_{cohort}.parquet"
        df = pd.read_parquet(path)
        # NOTE: the plain `risk_bad` column in these parquets is NOT the final
        # v3-model score reported in the manuscript (it does not reproduce
        # Table_1_external_validation_v3.csv's Brier values). `risk_bad_v3` is
        # the column that exactly reproduces Table 1 / local_recalibration
        # summary Brier scores for every cohort, so it is used here -- this
        # keeps the reliability diagrams consistent with the scalar Brier/ECE
        # already reported in Supp Table 7B.
        risk = df["risk_bad_v3"].to_numpy(dtype=float)
        gt = df["gt_bad"].to_numpy(dtype=float)
        n_total = len(df)

        bins = build_bins(risk, gt)
        bins.insert(0, "cohort", cohort)
        all_rows.append(bins)

        b = brier(risk, gt)
        e = ece(bins, n_total)
        # mean calibration gap signed (predicted - observed), weighted
        w = bins["n"] / n_total
        signed_gap = float((w * (bins["mean_predicted_risk"] - bins["observed_frequency"])).sum())
        summary_rows.append({
            "cohort": cohort,
            "n": n_total,
            "n_bins_after_merge": len(bins),
            "brier": b,
            "ece_weighted": e,
            "mean_signed_gap_pred_minus_obs": signed_gap,
            "direction": "over-confident (pred > obs)" if signed_gap > 0.005 else (
                "under-confident (pred < obs)" if signed_gap < -0.005 else "well-calibrated on average"),
        })
        print(f"{cohort}: n={n_total}, bins_after_merge={len(bins)}, Brier={b:.4f}, ECE={e:.4f}, signed_gap={signed_gap:+.4f}")

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_csv(OUT_DATA, index=False)
    OUT_TABLE_COPY.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_TABLE_COPY, index=False)
    print(f"\nSaved bin-level table to {OUT_DATA}\nand {OUT_TABLE_COPY}")

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary (signed gap = mean(pred - obs), weighted by bin n) ===")
    print(summary_df.to_string(index=False))

    summary_path = DATA_DIR / "calibration_reliability_summary_v3.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(BASE / "results" / "generated" / "calibration_reliability_summary_v3.csv", index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
