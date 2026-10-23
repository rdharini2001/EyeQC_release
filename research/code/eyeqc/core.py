"""EyeQC feature extraction and contract-based inference.

The feature extractor is shared by model development, validation, batch
experiments, and the GUI. Model fitting lives in :mod:`eyeqc.development` so
inference cannot silently retrain or alter a frozen contract.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from typing import Mapping
import cv2
import numpy as np
from scipy import ndimage as ndi

FEATURES = (
    "focus",
    "detail_uniformity",
    "local_readability",
    "illumination_uniformity",
    "exposure",
    "contrast",
    "field_coverage",
    "artifact_burden",
    "clipping",
    "color_balance",
)

DISPLAY_NAMES = {
    "focus": "Focus",
    "detail_uniformity": "Detail uniformity",
    "local_readability": "Local readability",
    "illumination_uniformity": "Illumination uniformity",
    "exposure": "Exposure",
    "contrast": "Contrast",
    "field_coverage": "Field coverage",
    "artifact_burden": "Artifact burden",
    "clipping": "Clipping",
    "color_balance": "Color balance",
}

ACTION_RULES = {
    "focus": "Refocus the camera; stabilize fixation and reacquire.",
    "detail_uniformity": "Check motion, media haze and localized blur.",
    "local_readability": "Recenter and reacquire to restore readable retinal area.",
    "illumination_uniformity": "Realign the camera and reduce shadow or arc artifacts.",
    "exposure": "Adjust flash/exposure or pupil conditions and reacquire.",
    "contrast": "Check media opacity, veiling glare and lens cleanliness.",
    "field_coverage": "Recenter the field and include the required retinal region.",
    "artifact_burden": "Clean optics; remove lashes/reflections and reacquire.",
    "clipping": "Reduce saturated highlights or crushed shadows.",
    "color_balance": "Check white balance and channel integrity.",
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
    d = lo - x if x < lo else x - hi
    return float(np.exp(-0.5 * (d / max(soft, 1e-06)) ** 2))

def _safe_percentile(x: np.ndarray, q: float, default: float=0.0) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else default

def _weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return np.nan
    values, weights = (values[ok], weights[ok])
    order = np.argsort(values)
    values, weights = (values[order], weights[order])
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return float(np.interp(q, cumulative, values))

def _resize_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError('expected an HxWx3 or HxWx4 image')
    a = np.ascontiguousarray(a[..., :3].astype(np.uint8))
    h, w = a.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        a = cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return a

def _keep_largest_components(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    keep = np.zeros_like(m)
    total = m.size
    for j in order[:3]:
        lab = j + 1
        if stats[lab, cv2.CC_STAT_AREA] >= max(32, 0.01 * total):
            keep[labels == lab] = 1
    if keep.sum() == 0:
        keep[labels == 1 + order[0]] = 1
    return keep.astype(bool)

def _retinal_support(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    sat = hsv[..., 1].astype(np.float32)
    aa = lab[..., 1].astype(np.float32) - 128
    bb = lab[..., 2].astype(np.float32) - 128
    chroma = np.sqrt(aa * aa + bb * bb)
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=max(1.5, 0.008 * min(h, w)))
    texture = np.abs(gray.astype(np.float32) - blur)
    mask = (gray > 5) & (gray < 250) & ((sat > 10) | (chroma > 7) | (texture > 2.5))
    k = max(3, int(round(min(h, w) * 0.012)))
    k += k % 2 == 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    mask = ndi.binary_fill_holes(_keep_largest_components(mask))
    if mask.mean() < 0.08:
        fallback = (gray > 8) & (gray < 245) & (texture > 1.5)
        fallback = cv2.morphologyEx(fallback.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        fallback = ndi.binary_fill_holes(_keep_largest_components(fallback))
        if fallback.mean() > mask.mean():
            mask = fallback
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = np.nan
    hull = np.zeros_like(mask, np.uint8)
    sx = sy = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        circularity = float(4 * np.pi * area / (per * per)) if per > 0 else np.nan
        cv2.drawContours(hull, [cv2.convexHull(cnt)], -1, 1, thickness=-1)
        _, _, bw, bh = cv2.boundingRect(cnt)
        sx, sy = (bw / max(w, 1), bh / max(h, 1))
    else:
        hull = mask.astype(np.uint8)
    erode = max(1, int(round(min(h, w) * 0.006)))
    inner = ndi.binary_erosion(mask, iterations=erode)
    if inner.sum() < 64:
        inner = mask.copy()
    edges = [mask[0].mean(), mask[-1].mean(), mask[:, 0].mean(), mask[:, -1].mean()]
    return {'mask': mask, 'inner': inner, 'hull': hull.astype(bool), 'retina_fraction': float(mask.mean()), 'circularity': circularity, 'borders_touched': int(sum((x > 0.2 for x in edges))), 'cropped_or_zoomed': bool(sum((x > 0.2 for x in edges)) >= 2 or mask.mean() > 0.78), 'bbox_span_x': float(sx), 'bbox_span_y': float(sy)}

def _normalize_green(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = rgb[..., 1].astype(np.float32)
    vals = g[mask]
    if vals.size < 32:
        return g.astype(np.uint8)
    lo, hi = np.percentile(vals, [2, 98])
    gn = np.clip((g - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gn)

def _illumination_score(gray: np.ndarray, inner: np.ndarray) -> tuple[float, float]:
    h, w = gray.shape
    low = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=max(3, 0.05 * min(h, w)))
    yy, xx = np.where(inner)
    if len(xx) < 100:
        return (0.0, np.nan)
    step = max(1, len(xx) // 6000)
    xs = xx[::step] / max(w - 1, 1) * 2 - 1
    ys = yy[::step] / max(h - 1, 1) * 2 - 1
    z = low[yy[::step], xx[::step]]
    X = np.column_stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys])
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ coef
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    dr = np.percentile(z, 95) - np.percentile(z, 5)
    norm = float(mad / max(dr, 20))
    return (float(np.exp(-0.5 * (norm / 0.2) ** 2)), norm)

def _field_coverage(mask: np.ndarray, support: Mapping) -> tuple[float, dict]:
    h, w = mask.shape
    y0, y1 = (int(0.2 * h), int(0.8 * h))
    x0, x1 = (int(0.2 * w), int(0.8 * w))
    central = float(mask[y0:y1, x0:x1].mean())
    tiles = []
    for yi in range(4):
        for xi in range(4):
            a, b = (int(yi * h / 4), int((yi + 1) * h / 4))
            c, d = (int(xi * w / 4), int((xi + 1) * w / 4))
            tiles.append(mask[a:b, c:d].mean())
    grid = float(np.mean(np.asarray(tiles) > 0.25))
    span = float(min(support['bbox_span_x'], support['bbox_span_y']))
    score = 0.25 * _ramp(support['retina_fraction'], 0.16, 0.45) + 0.25 * _ramp(central, 0.45, 0.95) + 0.3 * _ramp(grid, 0.35, 0.78) + 0.2 * _ramp(span, 0.45, 0.85)
    return (float(np.clip(score, 0, 1)), {'central_retina_coverage': central, 'grid_retina_presence': grid})

def _local_readability(gray: np.ndarray, gnorm: np.ndarray, mask: np.ndarray, grid: int=6) -> tuple[dict, list]:
    h, w = gray.shape
    gx = cv2.Sobel(gnorm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gnorm, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy) / 255
    qs = []
    ws = []
    ds = []
    rows = []
    for yi in range(grid):
        for xi in range(grid):
            y0, y1 = (int(yi * h / grid), int((yi + 1) * h / grid))
            x0, x1 = (int(xi * w / grid), int((xi + 1) * w / grid))
            m = mask[y0:y1, x0:x1]
            n = int(m.sum())
            if n < max(32, int(0.2 * m.size)):
                continue
            lum = gray[y0:y1, x0:x1][m]
            gd = grad[y0:y1, x0:x1][m]
            p10, p50, p90 = [_safe_percentile(lum, q) for q in (10, 50, 90)]
            ext = float(np.mean((lum < 10) | (lum > 247)))
            exp = _band_score(p50, 30, 225, 28) * np.exp(-7 * ext)
            con = _ramp(p90 - p10, 12, 70)
            detail = _ramp(_safe_percentile(gd, 85), 0.025, 0.16)
            quality = float(np.clip((max(exp, 1e-06) * max(con, 1e-06) * max(detail, 1e-06)) ** (1 / 3), 0, 1))
            qs.append(quality)
            ws.append(n)
            ds.append(detail)
            rows.append({'yi': yi, 'xi': xi, 'quality': quality, 'exposure': exp, 'contrast': con, 'detail': detail, 'weight': n})
    if not qs:
        return ({'local_readability': 0.0, 'readable_area_fraction': 0.0, 'detail_uniformity': 0.0, 'n_retinal_tiles': 0}, rows)
    q = np.asarray(qs)
    wt = np.asarray(ws)
    det = np.asarray(ds)
    mean = float(np.average(q, weights=wt))
    p20 = _weighted_quantile(q, wt, 0.2)
    readable = float(wt[q >= 0.4].sum() / wt.sum())
    local = 0.45 * mean + 0.3 * p20 + 0.25 * _ramp(readable, 0.45, 0.9)
    detail = 0.55 * float(np.average(det, weights=wt)) + 0.45 * _weighted_quantile(det, wt, 0.2)
    return ({'local_readability': float(np.clip(local, 0, 1)), 'readable_area_fraction': readable, 'detail_uniformity': float(np.clip(detail, 0, 1)), 'n_retinal_tiles': len(q)}, rows)

def _large_component_fraction(binary: np.ndarray, denom: int) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max(8, 0.0025 * denom):
            keep += area
    return float(keep / max(denom, 1))

def extract_features(rgb: np.ndarray, max_side: int=640) -> dict:
    """Return the ten frozen features plus auditable support and tile metadata."""
    t0 = time.perf_counter()
    rgb = _resize_rgb(rgb, max_side)
    support = _retinal_support(rgb)
    mask = support['mask']
    inner = support['inner']
    hull = support['hull']
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gnorm = _normalize_green(rgb, inner)
    lap = cv2.Laplacian(gnorm, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap[inner])) if inner.any() else 0
    gx = cv2.Sobel(gnorm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gnorm, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy) / 255
    grad90 = _safe_percentile(grad[inner], 90)
    med3 = cv2.medianBlur(gnorm, 3)
    resid = np.abs(gnorm.astype(np.float32) - med3.astype(np.float32))
    flat_thr = _safe_percentile(grad[inner], 35)
    flat = inner & (grad <= flat_thr)
    noise = _safe_percentile(resid[flat], 50) if flat.sum() > 32 else 0
    focus = float(np.sqrt(_ramp(lap_var, 18, 150) * _ramp(grad90, 0.035, 0.18)) * np.exp(-0.5 * (max(noise - 5, 0) / 8) ** 2))
    local, tiles = _local_readability(gray, gnorm, inner)
    illum, illum_raw = _illumination_score(gray, inner)
    vals = gray[inner] if inner.any() else gray.ravel()
    p5, p50, p95 = [_safe_percentile(vals, q) for q in (5, 50, 95)]
    under = float(np.mean(vals < 12))
    over = float(np.mean(vals > 245))
    exposure = float(_band_score(p50, 35, 220, 30) * np.exp(-7 * (under + over)))
    contrast = _ramp(p95 - p5, 25, 105)
    field, field_audit = _field_coverage(mask, support)
    sat = hsv[..., 1]
    spec = (gray > 245) & (sat < 22) & hull
    dark = (gray < 10) & hull & ~mask
    denom = max(int(hull.sum()), 1)
    glare = _large_component_fraction(spec, denom)
    dark_occ = _large_component_fraction(dark, denom)
    artifact_fraction = float(np.clip(glare + dark_occ + 0.3 * max(1 - local['readable_area_fraction'] - 0.35, 0), 0, 1))
    artifact = float(np.exp(-10 * artifact_fraction))
    pix = rgb[inner] if inner.any() else rgb.reshape(-1, 3)
    clip_fraction = float(np.mean((pix <= 2) | (pix >= 253)))
    clipping = float(np.exp(-10 * clip_fraction))
    med = np.median(pix, axis=0)
    rg = float((med[0] + 1) / (med[1] + 1))
    gb = float((med[1] + 1) / (med[2] + 1))
    sat_med = float(np.median(sat[inner]) / 255) if inner.any() else 0
    color = float((_band_score(rg, 0.9, 3.4, 0.75) * _band_score(gb, 0.75, 3.8, 0.9) * _band_score(sat_med, 0.06, 0.92, 0.18)) ** (1 / 3))
    out = {
        "focus": focus,
        "detail_uniformity": local["detail_uniformity"],
        "local_readability": local["local_readability"],
        "illumination_uniformity": illum,
        "exposure": exposure,
        "contrast": contrast,
        "field_coverage": field,
        "artifact_burden": artifact,
        "clipping": clipping,
        "color_balance": color,
    }
    out.update(
        {
            "runtime_feature_sec": time.perf_counter() - t0,
            "retina_fraction": support["retina_fraction"],
            "circularity_audit": support["circularity"],
            "borders_touched": support["borders_touched"],
            "cropped_or_zoomed": support["cropped_or_zoomed"],
            "readable_area_fraction": local["readable_area_fraction"],
            "illumination_residual_raw": illum_raw,
            "glare_fraction_raw": glare,
            "dark_occlusion_fraction_raw": dark_occ,
            "artifact_fraction_raw": artifact_fraction,
            "clip_fraction_raw": clip_fraction,
            "tiles": tiles,
            "analysis_rgb": rgb,
            "support_mask": mask,
        }
    )
    out.update(field_audit)
    return out

# Scope gate: thresholds below were set from, and validated against, the
# five real validation cohorts (EyeQ-test, DRIMDB, FQS, MSHF-CFP, DeepDRiD) --
# see code_final/run_scope_gate_validation.py and
# results_final/tables/scope_gate_validation.csv for the full protocol and
# numbers. Do not change these without re-running that validation.
SCOPE_RETINA_FRACTION_FLOOR = 0.30
SCOPE_BLANK_EPS = 0.03

def scope_check(features: Mapping[str, float]) -> dict:
    """Flag inputs that do not look like a fundus photograph at all.

    This is deliberately separate from the ten quality axes scored by
    :class:`EyeQCModel`: a blurry or poorly-exposed *fundus* photo should
    still get a PASS/REVIEW/FAIL quality verdict, but a photo of a hand, a
    document scan, a solid-color frame, or random noise should not receive
    any quality verdict at all, because none of the ten axes are meaningful
    for it.

    Two rules, both derived from where real labelled images in the five
    validation cohorts actually sit (never a guessed threshold):

    1. ``retina_fraction < SCOPE_RETINA_FRACTION_FLOOR`` -- almost no part
       of the frame was detected as retina-like tissue. Across all five
       cohorts, every image with a valid "Good"/"Usable" ground-truth label
       had ``retina_fraction >= 0.408``; the handful of real images below
       that were all ground-truth "Reject" cases that turned out, on manual
       inspection, to be genuinely near-black failed captures. 0.30 sits in
       the empty gap between those two groups.
    2. A retina-like region fills the frame (so rule 1 does not fire) but
       has essentially zero internal contrast *and* zero locally-readable
       structure (``contrast < 0.03`` and ``readable_area_fraction < 0.03``).
       This catches frame-filling but content-free inputs such as a solid
       saturated color, which can produce a high ``retina_fraction`` (a
       uniform saturated color trips the same texture/chroma test used to
       detect retinal tissue) despite having no retinal structure at all.

    Neither rule uses ``circularity_audit`` as a hard threshold: a real,
    validly-labelled MSHF-CFP photo was found during validation with
    ``circularity_audit`` as low as 0.163 (an eyelash/lid-occluded border
    made the detected mask boundary jagged), which would have been a false
    positive for any plausible single circularity cutoff. ``circularity_audit``
    remains available as a diagnostic field but is intentionally not part of
    this gate.

    This is a conservative heuristic, not a general-purpose non-fundus image
    classifier: it reliably catches blank/near-empty/near-uniform frames but
    does **not** reliably catch an arbitrary photo, document scan, or random
    noise image that happens to fill the frame with texture and contrast --
    see the synthetic-image results in scope_gate_validation.csv for exactly
    which synthetic failure modes are and are not caught.

    Parameters
    ----------
    features:
        A mapping that includes at least ``retina_fraction``, ``contrast``,
        and ``readable_area_fraction``, as returned by :func:`extract_features`.

    Returns
    -------
    dict with keys ``in_scope`` (bool) and ``reason`` (str or None).
    """
    rf = float(features.get("retina_fraction", float("nan")))
    contrast = float(features.get("contrast", float("nan")))
    readable = float(features.get("readable_area_fraction", float("nan")))

    if not np.isfinite(rf):
        return {
            "in_scope": False,
            "reason": "retina_fraction unavailable (feature extraction failed or produced no finite value)",
        }

    if rf < SCOPE_RETINA_FRACTION_FLOOR:
        return {
            "in_scope": False,
            "reason": (
                f"retina-like area covers only {rf:.1%} of the frame "
                f"(< {SCOPE_RETINA_FRACTION_FLOOR:.0%}); likely near-black, "
                "near-white, severely cropped, or not a fundus photograph"
            ),
        }

    if np.isfinite(contrast) and np.isfinite(readable) and contrast < SCOPE_BLANK_EPS and readable < SCOPE_BLANK_EPS:
        return {
            "in_scope": False,
            "reason": (
                "a retina-like region fills the frame but shows no internal "
                f"contrast or readable structure (contrast={contrast:.3f}, "
                f"readable_area_fraction={readable:.3f}); likely a blank, "
                "uniform-color, or non-fundus image"
            ),
        }

    return {"in_scope": True, "reason": None}

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)

class EyeQCModel:
    """Inference-only EyeQC implementation backed by a versioned JSON contract."""

    def __init__(self, model_path: str | Path | None=None):
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / 'models' / 'eyeqc_model.json'
        self.path = path
        self.contract = json.loads(path.read_text(encoding='utf-8'))
        if tuple(self.contract['features']) != FEATURES:
            raise ValueError('model feature order does not match the validated extractor')
        self.mean = np.asarray(self.contract['standardizer_mean'], float)
        self.scale = np.asarray(self.contract['standardizer_scale'], float)
        self.coef = np.asarray(self.contract['logistic_coefficients_standardized'], float)
        self.intercept = float(self.contract['logistic_intercept'])
        if not self.mean.shape == self.scale.shape == self.coef.shape == (len(FEATURES),):
            raise ValueError('invalid model-contract dimensions')
        if not (np.isfinite(self.mean).all() and np.isfinite(self.scale).all() and np.isfinite(self.coef).all() and np.isfinite(self.intercept)):
            raise ValueError('model contract contains non-finite parameters')
        if np.any(self.scale <= 0):
            raise ValueError('standardizer scales must be positive')
        pass_t = float(self.contract['pass_risk_threshold'])
        fail_t = float(self.contract['fail_risk_threshold'])
        if not 0 <= pass_t < fail_t <= 1:
            raise ValueError('invalid PASS/FAIL risk thresholds')

    def score_features(self, features: Mapping[str, float]) -> dict:
        x = np.asarray([features[k] for k in FEATURES], float)
        if not np.isfinite(x).all():
            raise ValueError('all ten EyeQC features must be finite')
        if np.any((x < 0) | (x > 1)):
            raise ValueError('EyeQC feature values must lie in [0, 1]')
        z = (x - self.mean) / self.scale
        raw_logit = float(self.intercept + z @ self.coef)
        calibrated_logit = float(self.contract['platt_intercept'] + self.contract['platt_coefficient'] * raw_logit)
        risk = _sigmoid(calibrated_logit)
        pass_t = float(self.contract['pass_risk_threshold'])
        fail_t = float(self.contract['fail_risk_threshold'])
        verdict = 'FAIL' if risk >= fail_t else 'PASS' if risk <= pass_t else 'REVIEW'
        contributions = np.asarray(self.contract['platt_coefficient']) * self.coef * z
        contribution_map = {k: float(v) for k, v in zip(FEATURES, contributions)}
        weakest = min(FEATURES, key=lambda k: features[k])
        dist = min(abs(risk - pass_t), abs(risk - fail_t))
        band = max(fail_t - pass_t, 1e-06)
        threshold_ambiguity = float(np.exp(-dist / band))
        axis_conflict = float(np.std(x))
        uncertainty = float(np.clip(0.75 * threshold_ambiguity + 0.25 * axis_conflict, 0, 1))
        return {
            "model_version": self.contract["version"],
            "feature_hash": self.contract["feature_hash"],
            "risk_bad": risk,
            "quality_score": 1 - risk,
            "verdict": verdict,
            "raw_logit": raw_logit,
            "calibrated_logit": calibrated_logit,
            "base_calibrated_logit": float(
                self.contract["platt_intercept"]
                + self.contract["platt_coefficient"] * self.intercept
            ),
            "axis_contributions": contribution_map,
            "weakest_axis": weakest,
            "suggested_action": ACTION_RULES[weakest],
            "threshold_ambiguity": threshold_ambiguity,
            "axis_conflict": axis_conflict,
            "workflow_uncertainty": uncertainty,
            "uncertainty_note": (
                "Threshold proximity plus axis conflict; OOD status requires "
                "a locally fitted reference distribution."
            ),
        }

    def analyze(self, rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        features = extract_features(rgb, int(self.contract.get('max_side', 640)))
        scored = self.score_features(features)
        scored['runtime_end_to_end_sec'] = time.perf_counter() - t0
        scored['features'] = features
        return scored
