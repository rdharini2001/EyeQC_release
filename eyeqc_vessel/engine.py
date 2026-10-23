"""Vessel-structure inference engine: lazy model loading, single-image CPU
forward pass, and the ten-number interpretable summary of the resulting
artery/vein/vessel-union probability maps.

Internal engineering note: never surface "RRWNet", "deep learning", "neural
network", or "segmentation model" to end users — see ``webapp/server.py``'s
``/api/vessel`` endpoint for the clinical-language framing this module
feeds into.

CPU-only, same convention as ``run_throughput_v3.py``: CUDA_VISIBLE_DEVICES
is forced empty before the first ``import torch`` in this process, so the
model never silently grabs a GPU even on a machine that has one (the
production webapp is deployed CPU-only).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # must precede the first `import torch`

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from .model import RRWNet  # noqa: E402
from .preprocessing import enhance_image  # noqa: E402

HF_MODEL_ID = "j-morano/rrwnet-rite"
INPUT_SIDE = 576
ITERATIONS = 5
VESSEL_THRESHOLD = 0.50

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def is_loaded(device: str = "cpu") -> bool:
    return device in _MODEL_CACHE


def load_rrwnet(device: str = "cpu") -> torch.nn.Module:
    """Lazily load and cache the vessel-structure model for `device`.

    Weights are not vendored in this repo: they are pulled once from the
    public Hugging Face Hub repo `j-morano/rrwnet-rite` via
    `PyTorchModelHubMixin.from_pretrained`, then cached locally by
    `huggingface_hub` (same lazy first-use download pattern as the QC
    backbone's `torch.hub.load` in `eyeqc_v2/backbone.py`). Raises on
    failure (e.g. no network reachable) — callers that need fail-soft
    behaviour (server startup warm-up, the request handler) catch it
    themselves so the rest of the app keeps working.
    """
    if device in _MODEL_CACHE:
        return _MODEL_CACHE[device]

    from huggingface_hub import PyTorchModelHubMixin

    class _RRWNetHub(RRWNet, PyTorchModelHubMixin):
        def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=ITERATIONS):
            super().__init__(input_ch=input_ch, output_ch=output_ch, base_ch=base_ch, iterations=iterations)

    model = _RRWNetHub.from_pretrained(HF_MODEL_ID)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _MODEL_CACHE[device] = model
    return model


def _resize_to_canvas(rgb: np.ndarray, side: int = INPUT_SIDE) -> tuple[np.ndarray, dict]:
    """Letterbox `rgb` into a `side`x`side` canvas, preserving aspect ratio."""
    h, w = rgb.shape[:2]
    scale = side / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp)
    y0, x0 = (side - nh) // 2, (side - nw) // 2
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, {"y0": y0, "x0": x0, "nh": nh, "nw": nw, "orig_h": h, "orig_w": w}


def _estimate_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Coarse retinal-tissue vs. background ROI: bright/near-black cutoff,
    morphological close, keep the largest connected component, fill holes.
    Crop-tolerant by design (does not assume a centered fundus circle)."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mx = rgb.max(axis=2)
    mask = (mx > 18) & (gray > 8)

    k = max(3, int(round(min(h, w) * 0.012)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest

    return ndi.binary_fill_holes(mask)


def predict_vessels(rgb: np.ndarray, device: str = "cpu") -> dict:
    """Run one forward pass and return artery/vein/vessel-union probability
    maps both in the model's native 576x576 canvas space (used for the
    interpretable feature summary) and resampled back into the original
    photograph's coordinate space (used for overlay rendering).
    """
    model = load_rrwnet(device)

    canvas, meta = _resize_to_canvas(rgb, INPUT_SIDE)
    y0, x0, nh, nw = meta["y0"], meta["x0"], meta["nh"], meta["nw"]

    resized_rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if max(rgb.shape[:2]) > INPUT_SIDE else cv2.INTER_CUBIC)
    tissue_small = _estimate_tissue_mask(resized_rgb)
    roi_canvas = np.zeros((INPUT_SIDE, INPUT_SIDE), dtype=np.uint8)
    roi_canvas[y0:y0 + nh, x0:x0 + nw] = tissue_small.astype(np.uint8)

    degenerate = int(roi_canvas.sum()) < 64
    if degenerate:
        enhanced = np.zeros_like(canvas, dtype=np.float32)
        eroded_roi = roi_canvas.astype(np.float32)
    else:
        enhanced, eroded_roi = enhance_image(canvas, roi_canvas, int_format=False)
        enhanced = np.asarray(enhanced, dtype=np.float32)
        eroded_roi = np.asarray(eroded_roi, dtype=np.float32)
        if not np.isfinite(enhanced).all():
            degenerate = True
            enhanced = np.zeros_like(canvas, dtype=np.float32)

    tensor = torch.from_numpy(enhanced.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(eroded_roi[None, None, ...]).float().to(device)

    with torch.inference_mode():
        pred = model(tensor)[-1]
        prob = torch.sigmoid(pred) * mask_tensor

    prob_np = prob.detach().cpu().numpy()[0].astype(np.float32)  # (3, side, side): artery, vein, vessel-union
    roi_np = mask_tensor.detach().cpu().numpy()[0, 0].astype(np.float32)
    if degenerate:
        prob_np[...] = 0.0

    h, w = meta["orig_h"], meta["orig_w"]
    crop = prob_np[:, y0:y0 + nh, x0:x0 + nw]
    artery = cv2.resize(crop[0], (w, h), interpolation=cv2.INTER_LINEAR)
    vein = cv2.resize(crop[1], (w, h), interpolation=cv2.INTER_LINEAR)
    vessel = cv2.resize(crop[2], (w, h), interpolation=cv2.INTER_LINEAR)
    roi_crop = roi_np[y0:y0 + nh, x0:x0 + nw]
    roi = cv2.resize(roi_crop, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5

    return {
        "artery": artery.astype(np.float32),
        "vein": vein.astype(np.float32),
        "vessel": vessel.astype(np.float32),
        "roi": roi,
        "prob_canvas": prob_np,
        "roi_canvas": roi_np,
        "degenerate": degenerate,
        "meta": meta,
    }


def _ramp(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if hi <= lo:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def _band_score(x: float, lo: float, hi: float, soft: float) -> float:
    if not np.isfinite(x):
        return 0.0
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(np.exp(-0.5 * (d / max(soft, 1e-6)) ** 2))


DEGENERATE_FEATURES = {
    "rrwnet_vessel_mass": 0.0,
    "rrwnet_vessel_density_05": 0.0,
    "rrwnet_vessel_p95": 0.0,
    "rrwnet_vessel_spread": 0.0,
    "rrwnet_av_separation": 0.0,
    "rrwnet_artery_mass": 0.0,
    "rrwnet_vein_mass": 0.0,
    "rrwnet_av_balance": 0.0,
    "rrwnet_av_mass_ratio": float("nan"),
    "rrwnet_vascular_gradability": 0.0,
}


def summarize_vessel_features(probs: np.ndarray, roi: np.ndarray) -> dict:
    """Turn the three canvas-space probability maps (artery, vein,
    vessel-union — `probs`, shape (3, H, W)) into ten interpretable scalar
    features, using the ROI mask (`roi`, shape (H, W)) that was already
    baked into `probs` by `predict_vessels`. Ported verbatim (thresholds and
    all) from the validation notebook's `summarize_rrwnet_probabilities`.
    """
    pa, pvein, pbv = (np.asarray(probs[i], dtype=np.float32) for i in range(3))
    m = np.asarray(roi) > 0.5
    if m.sum() < 64:
        return dict(DEGENERATE_FEATURES)

    vals = pbv[m]
    vessel_mass = float(vals.mean())
    vessel_density = float(np.mean(vals >= VESSEL_THRESHOLD))
    vessel_p95 = float(np.percentile(vals, 95))

    h, w = m.shape
    tile_present = []
    for yi in range(4):
        for xi in range(4):
            y1, y2 = int(yi * h / 4), int((yi + 1) * h / 4)
            x1, x2 = int(xi * w / 4), int((xi + 1) * w / 4)
            tm = m[y1:y2, x1:x2]
            if tm.mean() < 0.10:
                continue
            tile_p = pbv[y1:y2, x1:x2][tm]
            tile_present.append(float(tile_p.mean() >= 0.03) if tile_p.size else 0.0)
    vessel_spread = float(np.mean(tile_present)) if tile_present else 0.0

    vessel_region = m & (pbv >= 0.35)
    if vessel_region.sum() >= 16:
        weights = pbv[vessel_region]
        av_sep = float(np.sum(np.abs(pa[vessel_region] - pvein[vessel_region]) * weights) / max(weights.sum(), 1e-8))
    else:
        av_sep = 0.0

    artery_mass = float(np.mean((pa * pbv)[m]))
    vein_mass = float(np.mean((pvein * pbv)[m]))
    av_balance = float(2 * min(artery_mass, vein_mass) / max(artery_mass + vein_mass, 1e-8))
    av_mass_ratio = float(artery_mass / max(vein_mass, 1e-8))

    # Frozen, deliberately broad biologic priors: these summarize the deep
    # map, they do not detect vessels themselves.
    density_score = _band_score(vessel_density, 0.012, 0.22, 0.08)
    confidence_score = _ramp(vessel_p95, 0.50, 0.90)
    spread_score = _ramp(vessel_spread, 0.20, 0.80)
    av_separation_score = _ramp(av_sep, 0.15, 0.65)
    vascular_gradability = float(np.clip(
        (density_score * confidence_score * spread_score * av_separation_score) ** 0.25, 0, 1
    ))

    return {
        "rrwnet_vessel_mass": vessel_mass,
        "rrwnet_vessel_density_05": vessel_density,
        "rrwnet_vessel_p95": vessel_p95,
        "rrwnet_vessel_spread": vessel_spread,
        "rrwnet_av_separation": av_sep,
        "rrwnet_artery_mass": artery_mass,
        "rrwnet_vein_mass": vein_mass,
        "rrwnet_av_balance": av_balance,
        "rrwnet_av_mass_ratio": av_mass_ratio,
        "rrwnet_vascular_gradability": vascular_gradability,
    }
