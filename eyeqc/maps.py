"""Spatial quality maps.

EyeQC scores local regions of the retina on three properties (exposure, contrast, detail) and combines them into a local quality
score.  The same formulas (:func:`eyeqc.core.region_scores`) are evaluated on two layouts:

``tile_grid``     the 6 x 6 grid that feeds the global ``local_readability`` and ``detail_uniformity`` measurements;
``quality_maps``  overlapping windows, giving a dense map that outlines where a score is low.

The maps describe image measurements. They are not attributions of the EyeQC risk score.

Command line::

    python -m eyeqc.maps photo.jpg --out photo_maps.png          # exposure and detail
    python -m eyeqc.maps photo.jpg --all                         # all four maps
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata

from .core import combine_scores, extract_features, quality_inputs, region_scores

MAP_KEYS = ("exposure", "contrast", "detail", "quality")
LOCALISING_KEYS = ("exposure", "detail")    # outline simulated defects well; contrast and combined often colour whole dim retinas
GRID = 6                    # tiles per side of the tile grid
WINDOW_DIVISOR = 8          # window side = image side / WINDOW_DIVISOR (chosen on development photographs)
STEPS_PER_WINDOW = 6        # window step = window side / STEPS_PER_WINDOW
MIN_RETINA_FRACTION = 0.5   # a window is scored only if at least this fraction of it is retina

# Display rule, identical for every map and photograph: colour shows how far a score falls below 1.
DISPLAY_FULL_DROP = 0.5     # a drop of this size or more is drawn in the strongest colour
DISPLAY_MIN_DROP = 0.02     # smaller drops are not coloured
DISPLAY_COLOURS = ("#FFD84D", "#F9962A", "#E8453C", "#A8106A")
# "spectral" style (interface): the whole retina is tinted from blue (score 1) through yellow to red (score 0).
SPECTRAL_COLOURS = ("#7A1E5A", "#C93756", "#F4793B", "#F8C95A", "#A7DCA0", "#3FA9C9", "#3558A6")
STYLES = ("drop", "spectral")


def quality_maps(features: Mapping, window_divisor: int = WINDOW_DIVISOR) -> dict[str, np.ndarray]:
    """Dense exposure, contrast, detail and quality maps.

    Parameters
    ----------
    features:
        Output of :func:`eyeqc.core.extract_features`.
    window_divisor:
        The window side is the image side divided by this number.

    Returns
    -------
    dict
        One ``float32`` array per key in :data:`MAP_KEYS`, at the analysis resolution, with scores in [0, 1] (1 = good)
        and ``NaN`` outside the retina.
    """
    inner = features["inner_mask"]
    gray, grad = quality_inputs(features["analysis_rgb"], inner)
    height, width = inner.shape
    win_h, win_w = max(8, height // window_divisor), max(8, width // window_divisor)
    step_y, step_x = max(2, win_h // STEPS_PER_WINDOW), max(2, win_w // STEPS_PER_WINDOW)
    tops = np.arange(-(win_h // 2), height - win_h // 2 + 1, step_y)
    lefts = np.arange(-(win_w // 2), width - win_w // 2 + 1, step_x)
    min_pixels = max(32, int(MIN_RETINA_FRACTION * win_h * win_w))

    scores = np.full((3, len(tops), len(lefts)), np.nan, np.float32)
    for i, top in enumerate(tops):
        rows = slice(max(top, 0), min(top + win_h, height))
        for j, left in enumerate(lefts):
            cols = slice(max(left, 0), min(left + win_w, width))
            retina = inner[rows, cols]
            if retina.sum() >= min_pixels:
                scores[:, i, j] = region_scores(gray[rows, cols][retina], grad[rows, cols][retina])

    scored = np.isfinite(scores[0])
    if not scored.any():
        return {key: np.full(inner.shape, np.nan, np.float32) for key in MAP_KEYS}
    # Retina pixels near the rim take the value of the nearest scored window.
    nearest = ndi.distance_transform_edt(~scored, return_distances=False, return_indices=True)
    # Bilinear interpolation between window centres.
    centre_x = (np.arange(width, dtype=np.float32) + 0.5 - (lefts[0] + win_w / 2)) / step_x
    centre_y = (np.arange(height, dtype=np.float32) + 0.5 - (tops[0] + win_h / 2)) / step_y
    map_x, map_y = np.meshgrid(centre_x.astype(np.float32), centre_y.astype(np.float32))

    maps = {}
    for key, layer in zip(MAP_KEYS[:3], scores):
        filled = layer[nearest[0], nearest[1]]
        maps[key] = cv2.remap(filled, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    maps["quality"] = combine_scores(maps["exposure"], maps["contrast"], maps["detail"]).astype(np.float32)
    return {key: np.where(inner, np.clip(value, 0, 1), np.nan).astype(np.float32) for key, value in maps.items()}


def pixel_auc(score_map: np.ndarray, defect: np.ndarray, max_pixels: int = 20_000) -> float:
    """Probability that a retinal pixel inside the defect has a lower score than one outside it (0.5 = chance)."""
    scored = np.isfinite(score_map)
    score, positive = score_map[scored], np.asarray(defect, bool)[scored]
    if score.size > max_pixels:
        keep = np.random.default_rng(0).choice(score.size, max_pixels, replace=False)
        score, positive = score[keep], positive[keep]
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(-score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def lowest_region(score_map: np.ndarray, tolerance: float = 0.02, ignore_above: float = 0.9) -> tuple[float, float] | None:
    """(row, column) inside the largest region of near-minimal score, or ``None`` if no score is clearly low."""
    if not np.isfinite(score_map).any() or np.nanmin(score_map) > ignore_above:
        return None
    low = (np.nan_to_num(score_map, nan=np.inf) <= np.nanmin(score_map) + tolerance).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(low)
    if count <= 1:
        return None
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    inside = ndi.distance_transform_edt(labels == largest)
    row, column = np.unravel_index(int(inside.argmax()), inside.shape)
    return float(row), float(column)


def tile_grid(features: Mapping, key: str = "quality") -> np.ndarray:
    """6 x 6 tile scores; ``NaN`` for tiles that contain too little retina to be scored."""
    grid = np.full((GRID, GRID), np.nan)
    for tile in features["tiles"]:
        grid[int(tile["yi"]), int(tile["xi"])] = tile[key]
    return grid


def lowest_tile_hit(grid: np.ndarray, defect_tiles: np.ndarray) -> float:
    """Whether the lowest-scoring retina tile lies in the defect.

    Only scored tiles are eligible. When several tiles share the lowest score the result is the fraction of them that lie
    in the defect, so the outcome never depends on the order in which tiles are visited.
    """
    scored = np.isfinite(grid)
    if not scored.any():
        return float("nan")
    lowest = scored & (grid <= np.nanmin(grid) + 1e-9)
    return float(np.asarray(defect_tiles, bool)[lowest].mean())


def overlay(features: Mapping, score_map: np.ndarray, style: str = "drop") -> np.ndarray:
    """RGBA image of a score map over the photograph, transparent outside the retina.

    ``style="drop"``      grey photograph; only low scores are coloured (used in the figures of the spatial analysis).
    ``style="spectral"``  every retinal pixel tinted from blue (good) to red (poor) (used in the interface).
    """
    from matplotlib.colors import LinearSegmentedColormap

    rgb, mask = features["analysis_rgb"], features["support_mask"]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    base = np.repeat(np.clip(gray * 0.95 + 18, 0, 255)[..., None], 3, axis=2) / 255
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    outline = (cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5) > 0.5).astype(np.float32)
    if style == "spectral":
        colours = LinearSegmentedColormap.from_list("eyeqc_spectral", SPECTRAL_COLOURS)
        score = np.nan_to_num(score_map, nan=1.0)
        blended = base * 0.30 + colours(np.clip(score, 0, 1))[..., :3] * 0.70
        return (np.dstack([blended, outline]) * 255).round().astype(np.uint8)
    drop = np.clip(np.nan_to_num(1 - score_map, nan=0.0), 0, 1)
    alpha = np.clip((drop - DISPLAY_MIN_DROP) / 0.08, 0, 1)[..., None] * 0.86
    colours = LinearSegmentedColormap.from_list("eyeqc_drop", DISPLAY_COLOURS)
    blended = base * (1 - alpha) + colours(np.clip(drop / DISPLAY_FULL_DROP, 0, 1))[..., :3] * alpha
    return (np.dstack([blended, outline]) * 255).round().astype(np.uint8)


def save_map_panel(image_path: str | Path, output_path: str | Path, keys: tuple[str, ...] = LOCALISING_KEYS) -> None:
    """Write a panel with the photograph and the requested dense maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .pipeline import read_rgb

    features = extract_features(read_rgb(image_path))
    maps = quality_maps(features)
    figure, axes = plt.subplots(1, 1 + len(keys), figsize=(3 * (1 + len(keys)), 3.3))
    photo = np.dstack([features["analysis_rgb"], features["support_mask"].astype(np.uint8) * 255])
    titles = ["Photograph"] + ["Combined" if key == "quality" else key.capitalize() for key in keys]
    for axis, title, image in zip(axes, titles, [photo] + [overlay(features, maps[key]) for key in keys]):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dense EyeQC quality maps for one fundus photograph.")
    parser.add_argument("--all", action="store_true", help="also draw the contrast and combined maps")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output PNG (default: <image name>_maps.png)")
    args = parser.parse_args(argv)
    output = args.out or args.image.with_name(f"{args.image.stem}_maps.png")
    save_map_panel(args.image, output, MAP_KEYS if args.all else LOCALISING_KEYS)
    print(output)


if __name__ == "__main__":
    main()
