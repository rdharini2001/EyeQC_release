"""Contrast-normalization preprocessing, ported from the vendored RRWNet
reference implementation and trimmed to the in-memory array path (the
original also supports reading images/masks from disk and batch folder
processing for the upstream training scripts; this package only ever
receives an already-decoded photograph from the webapp).

Source: third_party/rrwnet-03069bb673f87d4b3cee37df378e0f15e5ade280/preprocessing.py
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage.morphology import disk


def _crop_center(img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    y, x = img.shape[0], img.shape[1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def _to_0_1(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float64)
    interp_fun = interp1d([lo, hi], [0.0, 1.0])
    return interp_fun(img)


def enhance_image(img: np.ndarray, mask: np.ndarray, int_format: bool = False, disk_size: int = 5):
    """Local-contrast-normalize `img` within `mask` (upstream RRWNet's
    preprocessing recipe: zoom-and-crop to fill outside the ROI, band-pass
    via Gaussian-subtraction, then normalize by the in-ROI standard
    deviation). Both `img` (H, W, 3 uint8) and `mask` (H, W, {0,1}) are
    expected already decoded/resized by the caller.

    Returns (enhanced_image, eroded_mask).
    """
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if mask.ndim == 3:
        mask = np.sum(mask[:, :, :3], axis=2)

    img = img / 255
    mask = np.where(mask > 0.5, 1, 0)

    img_copy = img.copy()
    zoomed_image = Image.fromarray(np.uint8(img_copy * 255))
    zoomed_image = zoomed_image.resize(
        (int(img_copy.shape[1] * 1.15), int(img_copy.shape[0] * 1.15)),
        Image.BICUBIC,
    )
    zoomed_image = np.array(zoomed_image)
    zoomed_image = _crop_center(zoomed_image, img_copy.shape[1], img_copy.shape[0])
    zoomed_image = zoomed_image / 255.0

    kernel = disk(disk_size)
    mask = ndimage.binary_erosion(mask, kernel)
    mask = mask * 1.0

    img_copy[mask < 1.0] = 0.0

    mask_rgb = np.stack((mask, mask, mask), axis=2)

    composed_image = mask_rgb.copy()
    composed_image[mask_rgb == 1.0] = img_copy[mask_rgb == 1.0]
    composed_image[mask_rgb < 1.0] = zoomed_image[mask_rgb < 1.0]

    filtered_image = ndimage.gaussian_filter(composed_image, sigma=(10, 10, 0))

    subtracted_image = composed_image - filtered_image
    subtracted_image[mask_rgb < 1.0] = 0.0

    std = np.std(subtracted_image)
    enhanced_image = subtracted_image / std if std > 1e-8 else subtracted_image
    enhanced_image = _to_0_1(enhanced_image)
    enhanced_image[mask_rgb < 1.0] = 0.0

    mask_out = mask_rgb[:, :, 0]

    if int_format:
        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        mask_out = (mask_out * 255).astype(np.uint8)

    return enhanced_image, mask_out
