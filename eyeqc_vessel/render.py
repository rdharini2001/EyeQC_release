"""Visual outputs for the vessel-structure analysis endpoint.

Renders `engine.predict_vessels`'s artery/vein/vessel-union probability
maps into matplotlib figures using the same Nature-inspired muted palette
as the rest of the product (`eyeqc.figures.NATURE`), so this panel reads as
part of one visual system rather than a bolted-on tool with different
colors. Figures are returned as plain `plt.Figure` objects, ready to pass
through `webapp/server.py`'s existing `fig_png()` helper — nothing here
saves or displays anything itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from eyeqc.figures import NATURE, set_nature_style  # noqa: E402

from .engine import VESSEL_THRESHOLD  # noqa: E402


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _dimmed_grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    gray = np.mean(rgb, axis=2, keepdims=True).astype(np.float32)
    dimmed = np.clip(gray * 0.45 + 12, 0, 255)
    return np.repeat(dimmed, 3, axis=2)


def vessel_overlay_figure(rgb: np.ndarray, result: dict) -> plt.Figure:
    """Source photograph next to an artery/vein colour overlay: red for
    artery-type vessel signal, blue for vein-type vessel signal, painted
    over a dimmed grayscale of the original photograph so anatomy stays
    recognizable underneath."""
    set_nature_style()
    base_rgb = _dimmed_grayscale_rgb(rgb)

    artery_w = np.clip(result["artery"] * result["vessel"], 0, 1)[..., None]
    vein_w = np.clip(result["vein"] * result["vessel"], 0, 1)[..., None]
    artery_color = _hex_to_rgb(NATURE["red"])
    vein_color = _hex_to_rgb(NATURE["blue"])

    overlay = base_rgb.copy()
    overlay = overlay * (1 - artery_w) + artery_color * artery_w
    overlay = overlay * (1 - vein_w) + vein_color * vein_w
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    axes[0].imshow(rgb)
    axes[0].set_title("Source photograph")
    axes[1].imshow(overlay)
    axes[1].set_title("Vessel structure map")
    for axis in axes:
        axis.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=NATURE["red"], label="Artery-type vessel"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, color=NATURE["blue"], label="Vein-type vessel"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.03), frameon=False)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def vessel_skeleton_figure(rgb: np.ndarray, result: dict) -> plt.Figure:
    """Thresholded vessel-union mask reduced to a one-pixel-wide skeleton
    and drawn over a dimmed grayscale — a compact view of the vascular
    network's branching pattern and coverage that stays readable
    independent of local vessel width or contrast."""
    set_nature_style()
    vessel_mask = (result["vessel"] >= VESSEL_THRESHOLD) & result["roi"]
    skeleton = skeletonize(vessel_mask)

    base_rgb = _dimmed_grayscale_rgb(rgb).astype(np.uint8)
    overlay = base_rgb.copy()
    teal = _hex_to_rgb(NATURE["green"]).astype(np.uint8)  # NATURE["green"] == "#00A087", the teal in this palette
    overlay[skeleton] = teal

    coverage_pct = float(vessel_mask[result["roi"]].mean() * 100) if result["roi"].any() else 0.0

    figure, axis = plt.subplots(figsize=(5.8, 5.8))
    axis.imshow(overlay)
    axis.axis("off")
    axis.set_title(f"Vessel network map · {coverage_pct:.1f}% of the readable field", fontsize=10, fontweight="bold")
    figure.tight_layout()
    return figure
