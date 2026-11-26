"""
Task 1 (figure): reliability diagrams (small multiples) for the 5 EyeQC v3
external cohorts, built from calibration_reliability_v3.csv (produced by
calibration_reliability_v3.py). Matches eyeqc_figures.py's palette/style.
"""
from __future__ import annotations
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = REPOSITORY
BIN_CSV = BASE / "data_final" / "tables" / "calibration_reliability_v3.csv"
SUMMARY_CSV = BASE / "data_final" / "tables" / "calibration_reliability_summary_v3.csv"
OUT_PNG = BASE / "results" / "generated" / "Calibration_Reliability_Diagrams_v3.png"
OUT_PDF = BASE / "results" / "generated" / "Calibration_Reliability_Diagrams_v3.pdf"

PALETTE = {
    "navy": "#164A87",
    "blue": "#3F76B5",
    "teal": "#82C5C9",
    "salmon": "#EA706B",
    "yellow": "#E9C64F",
    "light_yellow": "#F1E17A",
    "purple": "#8A69A6",
    "grey": "#A0A7B0",
    "grid": "#DDE4EB",
    "ink": "#111111",
    "muted": "#5F6872",
    "white": "#FFFFFF",
}
COHORT_ORDER = ["DRIMDB", "EyeQ-test", "MSHF-CFP", "FQS", "DeepDRiD"]
COHORT_COLORS = {
    "DRIMDB": PALETTE["navy"],
    "EyeQ-test": PALETTE["blue"],
    "MSHF-CFP": PALETTE["teal"],
    "FQS": PALETTE["yellow"],
    "DeepDRiD": PALETTE["purple"],
}


def set_style():
    mpl.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
    })
    try:
        mpl.rcParams["font.family"] = "Arimo"
        import matplotlib.font_manager as fm
        if "Arimo" not in {f.name for f in fm.fontManager.ttflist}:
            mpl.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        mpl.rcParams["font.family"] = "DejaVu Sans"


def style_axis(ax):
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(axis="both", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    set_style()
    bins = pd.read_csv(BIN_CSV)
    summ = pd.read_csv(SUMMARY_CSV).set_index("cohort")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), constrained_layout=True)
    axes_flat = axes.flatten()

    for i, cohort in enumerate(COHORT_ORDER):
        ax = axes_flat[i]
        sub = bins[bins.cohort == cohort].sort_values("bin_lo")
        color = COHORT_COLORS[cohort]

        # perfect calibration diagonal
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.1, color=PALETTE["grey"], zorder=1)

        x = sub["mean_predicted_risk"].to_numpy()
        y = sub["observed_frequency"].to_numpy()
        n = sub["n"].to_numpy()
        sizes = 25 + 180 * (n / n.max())

        ax.plot(x, y, color=color, linewidth=1.6, zorder=2, alpha=0.85)
        ax.scatter(x, y, s=sizes, color=color, edgecolor="white", linewidth=0.7, zorder=3)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        style_axis(ax)

        row = summ.loc[cohort]
        ax.set_title(
            f"{cohort}  (n={int(row['n'])})\nBrier={row['brier']:.3f}, ECE={row['ece_weighted']:.3f}",
            fontsize=10, color=PALETTE["ink"]
        )
        ax.set_xlabel("Mean predicted risk (bin)", fontsize=9)
        ax.set_ylabel("Observed frequency (gt_bad=1)", fontsize=9)

    # 6th panel: legend / marker-size key + brief annotation
    ax = axes_flat[5]
    ax.axis("off")
    ax.text(0.0, 0.95, "Reliability diagrams: risk_bad_v3 vs gt_bad", fontsize=10.5,
             fontweight="bold", color=PALETTE["ink"], transform=ax.transAxes, va="top")
    ax.text(0.0, 0.84,
            "10 equal-width bins on predicted risk [0,1];\n"
            "bins with n<20 merged into a neighboring bin\n"
            "(DRIMDB and MSHF-CFP have some merges due\n"
            "to small/imbalanced cohort size).\n\n"
            "Marker size \u221d bin sample size n.\n"
            "Dashed line = perfect calibration.",
            fontsize=8.8, color=PALETTE["muted"], transform=ax.transAxes, va="top")
    ax.text(0.0, 0.42,
            "Key finding: DeepDRiD is systematically\n"
            "UNDER-confident, concentrated at LOW\n"
            "predicted risk (the PASS band) \u2014 images\n"
            "scored risk\u22480.03 are actually poor-quality\n"
            "28% of the time (vs 0.8-9% in EyeQ-test/FQS).\n"
            "This is a specific, clinically relevant failure\n"
            "mode, not just a worse scalar ECE.",
            fontsize=8.8, color=PALETTE["ink"], transform=ax.transAxes, va="top")

    fig.suptitle("Supplementary Figure: Per-cohort calibration reliability diagrams (EyeQC v3)",
                 fontsize=12.5, color=PALETTE["ink"])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.12)
    print(f"Saved {OUT_PNG}\nSaved {OUT_PDF}")


if __name__ == "__main__":
    main()
