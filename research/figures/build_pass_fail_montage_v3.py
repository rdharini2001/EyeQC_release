"""Pass/fail case montage across every validation cohort — v3 hybrid model.

One row per dataset — EyeQ-test, DRIMDB, FQS, MSHF-CFP, DeepDRiD — each
showing real PASS exemplars next to real FAIL exemplars (diversified across
which axis drove the FAIL), every photograph carrying its local defect-map
heatmap overlay. Verdicts/risk scores come straight from each cohort's own
`image_level_v3_*.parquet` table (the same real, already-computed v3 scores
used throughout the manuscript) — the ten-axis features that drive the
overlay and the "weakest axis" label are model-independent (identical for
v1/v2/v3, since they come from the hand-crafted feature extractor, not the
decision head), so they're read straight off each row rather than recomputed.
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
ROUTE_COLORS = {"PASS": NATURE["green"], "REVIEW": "#E6A700", "FAIL": NATURE["red"]}
N_PASS = 3
N_FAIL = 3
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
    """Blend a magma defect-map over the photograph, with per-tile opacity
    weighted by badness so near-good tiles stay clean and only genuinely
    poor tiles are tinted."""
    grid = tile_quality_map(features)
    badness = np.clip(1 - grid, 0, 1)
    heatmap = np.array(plt.get_cmap("magma")(np.kron(badness, np.ones((40, 40))))[..., :3] * 255, dtype=np.uint8)
    heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
    local_alpha = alpha * cv2.resize(np.kron(badness**1.6, np.ones((40, 40))), (rgb.shape[1], rgb.shape[0]))[..., None]
    return np.clip((1 - local_alpha) * rgb.astype(float) + local_alpha * heatmap.astype(float), 0, 255).astype(np.uint8)


def load_cohort_frame(cohort: str) -> pd.DataFrame:
    frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
    frame = frame[frame["image_path"].map(lambda p: Path(str(p)).exists())].copy()
    frame["quality_score_v3"] = 1 - frame["risk_bad_v3"]
    return frame


def pick_examples(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[list, list]:
    pass_pool = frame[frame["verdict_v3"].eq("PASS")].sort_values("risk_bad_v3")
    pass_examples = list(pass_pool.head(N_PASS * 4).sample(n=min(N_PASS, len(pass_pool)), random_state=int(rng.integers(1e6))).itertuples())

    fail_pool = frame[frame["verdict_v3"].eq("FAIL")].sort_values("risk_bad_v3", ascending=False)
    fail_examples, seen_axes = [], set()
    for row in fail_pool.itertuples():
        axis = getattr(row, "weakest_axis", None)
        if axis in seen_axes:
            continue
        fail_examples.append(row)
        seen_axes.add(axis)
        if len(fail_examples) >= N_FAIL:
            break
    if len(fail_examples) < N_FAIL:
        for row in fail_pool.itertuples():
            if len(fail_examples) >= N_FAIL:
                break
            if row not in fail_examples:
                fail_examples.append(row)
    return pass_examples, fail_examples[:N_FAIL]


def features_of(rgb: np.ndarray) -> dict:
    """Live-extracted features (needed for the tile-level heatmap grid,
    which isn't persisted in the results table) rather than the flat scalar
    columns already stored per row."""
    return extract_features(rgb)


def draw_cell(ax, overlay: np.ndarray, border_color: str, verdict: str, caption: str) -> None:
    ax.imshow(overlay)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3.0)
    ax.set_title(verdict, fontsize=9.6, fontweight="bold", color=border_color, pad=3)
    ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, color="#333333")


def main() -> None:
    set_style()
    rng = np.random.default_rng(RNG_SEED)

    validation = pd.read_csv(TABLES / "Table_1_external_validation_v3.csv").set_index("cohort")

    n_cols = N_PASS + N_FAIL
    fig, axes = plt.subplots(len(COHORTS), n_cols, figsize=(2.65 * n_cols, 2.95 * len(COHORTS)))

    fig.suptitle(
        "PASS and FAIL cases across every validation cohort — EyeQC v3 (hybrid model), with the local defect map overlaid",
        fontsize=17,
        fontweight="bold",
        x=0.01,
        ha="left",
        y=0.995,
    )
    fig.text(
        0.01,
        0.975,
        "Each photograph's verdict and risk score come from the v3 hybrid model (ten interpretable measures plus a frozen "
        "visual embedding). The overlay tints only the tiles that actually drove the ten-axis read, brighter where the "
        "local read is worse — this map is identical for every EyeQC version, since it doesn't depend on the decision head. "
        "FAIL examples are diversified across which quality axis triggered the route.",
        fontsize=10.5,
        color="#4a5568",
        ha="left",
    )

    for row, cohort in enumerate(COHORTS):
        frame = load_cohort_frame(cohort)
        pass_examples, fail_examples = pick_examples(frame, rng)

        stats = validation.loc[cohort]
        row_label = f"{cohort}\nN={int(stats['N']):,}\nAUROC={stats['AUROC']:.3f}"
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

        for col, row_obj in enumerate(pass_examples):
            rgb = read_rgb(row_obj.image_path)
            overlay = overlay_heatmap(rgb, features_of(rgb))
            caption = f"quality {row_obj.quality_score_v3 * 100:.0f}/100"
            draw_cell(axes[row, col], overlay, ROUTE_COLORS["PASS"], "PASS", caption)
        for col in range(len(pass_examples), N_PASS):
            axes[row, col].axis("off")

        for col, row_obj in enumerate(fail_examples):
            rgb = read_rgb(row_obj.image_path)
            overlay = overlay_heatmap(rgb, features_of(rgb))
            axis_label = DISPLAY_NAMES.get(getattr(row_obj, "weakest_axis", ""), "—")
            caption = f"{axis_label}-driven · risk {row_obj.risk_bad_v3:.2f}"
            draw_cell(axes[row, N_PASS + col], overlay, ROUTE_COLORS["FAIL"], "FAIL", caption)
        for col in range(len(fail_examples), N_FAIL):
            axes[row, N_PASS + col].axis("off")

    for col in range(n_cols):
        label = "PASS examples" if col < N_PASS else "FAIL examples"
        color = ROUTE_COLORS["PASS"] if col < N_PASS else ROUTE_COLORS["FAIL"]
        if col in (0, N_PASS):
            axes[0, col].text(
                0.0,
                1.28,
                label,
                transform=axes[0, col].transAxes,
                fontsize=12.5,
                fontweight="bold",
                color=color,
                ha="left",
            )

    fig.tight_layout(rect=(0.10, 0.0, 1, 0.955))
    fig.subplots_adjust(wspace=0.08, hspace=0.55)

    stub = FIGDIR / "Pass_Fail_Case_Montage_v3"
    fig.savefig(f"{stub}.png", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{stub}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {stub}.png / .pdf")


if __name__ == "__main__":
    main()
