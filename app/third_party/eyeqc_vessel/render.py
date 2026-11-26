"""Images shown for the vessel analysis (plain RGB arrays on a dark background)."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .engine import VESSEL_THRESHOLD

ARTERY = np.array([244, 104, 108], np.float32)
VEIN = np.array([73, 160, 255], np.float32)
SKELETON = (101, 219, 193)


def _dimmed(rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    base = np.repeat(np.clip(gray * 0.45 + 10, 0, 255)[..., None], 3, axis=2)
    base[~roi] = 0
    return base


def artery_vein_overlay(rgb: np.ndarray, result: dict) -> np.ndarray:
    """Artery-type signal in red and vein-type signal in blue over the dimmed photograph."""
    image = _dimmed(rgb, result["roi"])
    for probability, colour in ((result["artery"], ARTERY), (result["vein"], VEIN)):
        weight = np.clip(probability * result["vessel"] * 1.15, 0, 1)[..., None]
        image = image * (1 - weight) + colour * weight
    return np.clip(image, 0, 255).astype(np.uint8)


def vessel_probability(result: dict) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(np.clip(result["vessel"] * 255, 0, 255)), cv2.COLORMAP_VIRIDIS)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat[~result["roi"]] = 0
    return heat


def skeleton_overlay(rgb: np.ndarray, result: dict) -> np.ndarray:
    """One-pixel centreline of the thresholded vessel mask, thickened for display."""
    mask = (result["vessel"] >= VESSEL_THRESHOLD) & result["roi"]
    lines = skeletonize(mask).astype(np.uint8)
    width = max(1, round(max(mask.shape) / 600))
    if width > 1:
        lines = cv2.dilate(lines, np.ones((width, width), np.uint8))
    image = _dimmed(rgb, result["roi"]).astype(np.uint8)
    image[lines > 0] = SKELETON
    return image
