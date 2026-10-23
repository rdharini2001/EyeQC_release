"""Borderline REVIEW-case montage across every validation cohort — v3 hybrid model.

Companion to `build_pass_fail_montage_v3.py`: instead of clear PASS vs. clear
FAIL, this shows real photographs from EyeQC's middle tier — "worth a second
look" — the calibrated decision band between the PASS and FAIL thresholds
where the model is least certain. Three cases per cohort, chosen to span the
REVIEW band (near-PASS edge, middle, near-FAIL edge) rather than clustering
at one end, each annotated with the axis driving the score and why it landed
in this band (threshold proximity vs. conflicting axes) rather than a clean
PASS or FAIL.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code_final"
sys.path.insert(0, str(CODE_ROOT))

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eyeqc.core import DISPLAY_NAMES, extract_features  # noqa: E402
from eyeqc.experiments import tile_quality_map  # noqa: E402
from eyeqc.figures import NATURE  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
FIGDIR = REPO_ROOT / "results_final" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
REVIEW_COLOR = "#B06000"
N_REVIEW = 4
RNG_SEED = 20260922


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 600,
            "font.family": "Arial",
            "font.size": 9.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def overlay_heatmap(rgb: np.ndarray, features: dict, alpha: float = 0.32) -> np.ndarray:
    grid = tile_quality_map(features)
    badness = np.clip(1 - grid, 0, 1)
    heatmap = np.array(plt.get_cmap("magma")(np.kron(badness, np.ones((40, 40))))[..., :3] * 255, dtype=np.uint8)
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    local_alpha = alpha * cv2.resize(np.kron(badness**1.6, np.ones((40, 40))), (rgb.shape[1], rgb.shape[0]))[..., None]
    return np.clip((1 - local_alpha) * rgb.astype(float) + local_alpha * heatmap.astype(float), 0, 255).astype(np.uint8)


def load_cohort_frame(cohort: str) -> pd.DataFrame:
    frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
    frame = frame[frame["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    return frame


def pick_review_examples(frame: pd.DataFrame, rng: np.random.Generator) -> list:
    """Span the REVIEW band from its PASS-adjacent edge to its FAIL-adjacent
    edge by quantile, rather than sampling uniformly (which tends to
    cluster near the band's denser middle)."""
    pool = frame[frame["verdict_v3"].eq("REVIEW")].sort_values("risk_bad_v3").reset_index(drop=True)
    if pool.empty:
        return []
    n = min(N_REVIEW, len(pool))
    quantile_positions = np.linspace(0.08, 0.92, n)
    idx = np.clip((quantile_positions * (len(pool) - 1)).round().astype(int), 0, len(pool) - 1)
    idx = sorted(set(idx.tolist()))
    while len(idx) < n:
        candidate = int(rng.integers(0, len(pool)))
        if candidate not in idx:
            idx.append(candidate)
    return list(pool.iloc[sorted(idx)[:n]].itertuples())


def features_of(rgb: np.ndarray) -> dict:
    """Live-extracted features (needed for the tile-level heatmap grid,
    which isn't persisted in the results table) rather than the flat scalar
    columns already stored per row."""
    return extract_features(rgb)


def band_reason(row_obj) -> str:
    ambiguity = getattr(row_obj, "threshold_ambiguity", float("nan"))
    conflict = getattr(row_obj, "axis_conflict", float("nan"))
    if ambiguity == ambiguity and conflict == conflict:
        return "threshold-proximity" if ambiguity >= conflict else "axis-conflict"
    return "borderline"


def draw_cell(ax, overlay: np.ndarray, caption_top: str, caption_bottom: str) -> None:
    ax.imshow(overlay)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(REVIEW_COLOR)
        spine.set_linewidth(3.0)
    ax.set_title(caption_top, fontsize=9.0, fontweight="bold", color=REVIEW_COLOR, pad=3)
    ax.text(0.5, -0.06, caption_bottom, transform=ax.transAxes, ha="center", va="top", fontsize=7.6, color="#333333")


def main() -> None:
    set_style()
    rng = np.random.default_rng(RNG_SEED)

    validation = pd.read_csv(TABLES / "Table_1_external_validation_v3.csv").set_index("cohort")

    fig, axes = plt.subplots(len(COHORTS), N_REVIEW, figsize=(2.75 * N_REVIEW, 2.95 * len(COHORTS)))

    fig.suptitle(
        "Borderline “worth a second look” cases across every validation cohort — EyeQC v3 (hybrid model)",
        fontsize=17,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.995,
    )
    fig.text(
        0.01,
        0.975,
        "Four real photographs per cohort, spanning the calibrated REVIEW band from its PASS-adjacent edge to its "
        "FAIL-adjacent edge (left to right). Each is labeled with the estimated risk, the axis most responsible for "
        "the score, and why the case landed in this band — the operating point was genuinely close (threshold-"
        "proximity) or two axes pulled in different directions (axis-conflict) — rather than a confident PASS or FAIL.",
        fontsize=10.3,
        color="#4a5568",
        ha="left",
    )

    for row, cohort in enumerate(COHORTS):
        frame = load_cohort_frame(cohort)
        examples = pick_review_examples(frame, rng)

        stats = validation.loc[cohort]
        review_frac = float(stats["REVIEW_fraction"]) * 100
        row_label = f"{cohort}\nN={int(stats['N']):,}\n{review_frac:.1f}% REVIEW"
        axes[row, 0].text(
            -0.42,
            0.5,
            row_label,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=NATURE["navy"],
            linespacing=1.6,
        )

        for col, row_obj in enumerate(examples):
            rgb = read_rgb(row_obj.image_path)
            overlay = overlay_heatmap(rgb, features_of(rgb))
            axis_label = DISPLAY_NAMES.get(getattr(row_obj, "weakest_axis", ""), "—")
            caption_top = f"risk {row_obj.risk_bad_v3:.2f}"
            caption_bottom = f"{axis_label} · {band_reason(row_obj)}"
            draw_cell(axes[row, col], overlay, caption_top, caption_bottom)
        for col in range(len(examples), N_REVIEW):
            axes[row, col].axis("off")

    axes[0, 0].text(
        0.0,
        1.28,
        "REVIEW examples, ordered by risk (PASS-adjacent → FAIL-adjacent)",
        transform=axes[0, 0].transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=REVIEW_COLOR,
        ha="left",
    )

    fig.tight_layout(rect=(0.10, 0.0, 1, 0.955))
    fig.subplots_adjust(wspace=0.08, hspace=0.55)

    stub = FIGDIR / "Review_Band_Case_Montage_v3"
    fig.savefig(f"{stub}.png", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{stub}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {stub}.png / .pdf")


if __name__ == "__main__":
    main()
