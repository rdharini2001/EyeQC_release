"""BRISQUE (Mittal, Moorthy & Bovik, 2012) feature extraction, implemented
directly from the algorithm description -- no external BRISQUE package is
used (see module docstring in run_brisque_baseline.py for why: the `brisque`
pip package throws on this venv's numpy/scipy, and `pyiqa` is avoided
because of its risky dependency footprint).

Reference: A. Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image
Quality Assessment in the Spatial Domain," IEEE Transactions on Image
Processing, 2012.

Algorithm, exactly as the paper describes it
---------------------------------------------
1. Grayscale the image.
2. Compute Mean Subtracted Contrast Normalized (MSCN) coefficients:
       MSCN(i,j) = (I(i,j) - mu(i,j)) / (sigma(i,j) + C)
   where mu/sigma are the local mean/std under a 7x7 Gaussian window
   (sigma=1.166, matching the original paper), C=1 avoids division by zero
   on flat (zero-variance) patches.
3. Fit a zero-mean Generalized Gaussian Distribution (GGD) to the MSCN
   coefficients -- 2 parameters (shape alpha, variance sigma^2) -- via the
   standard method-of-moments / lookup-table estimator used throughout the
   NR-IQA literature (Sharifi & Leon-Garcia-style shape-vs-kurtosis
   matching). This gives 2 features.
4. Form the 4 pairwise products of neighbouring MSCN coefficients
   (horizontal, vertical, main-diagonal, secondary-diagonal) and fit an
   Asymmetric GGD (AGGD, 4 parameters: shape, mean, left-variance,
   right-variance) to each via the analogous method-of-moments estimator.
   This gives 4 x 4 = 16 features.
5. Steps 2-4 give 18 features at the original resolution. Downsample the
   image by a factor of 2 (area interpolation) and repeat for 18 more
   features at the coarser scale.
6. Total: 36 features per image.

This module only produces the 36-dimensional feature vector. Mapping those
features to a quality score is deliberately NOT done with the original
paper's pretrained natural-image SVR (see run_brisque_baseline.py for the
justification) -- that mapping is fit separately, on EyeQC's own
development cohort, in run_brisque_baseline.py.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn

# Precompute the GGD/AGGD shape lookup tables once at import time: the
# method-of-moments estimator works by matching an observed moment ratio to
# the ratio implied by a grid of candidate shape parameters, then taking the
# closest match. A finer grid gives a more precise shape estimate.
_GAMMA_GRID = np.arange(0.2, 10.0 + 1e-9, 0.001)
_GGD_RATIO_TABLE = (gamma_fn(1.0 / _GAMMA_GRID) * gamma_fn(3.0 / _GAMMA_GRID)) / (
    gamma_fn(2.0 / _GAMMA_GRID) ** 2
)
_AGGD_RATIO_TABLE = (gamma_fn(2.0 / _GAMMA_GRID) ** 2) / (
    gamma_fn(1.0 / _GAMMA_GRID) * gamma_fn(3.0 / _GAMMA_GRID)
)

_WINDOW_SIZE = 7
_WINDOW_SIGMA = 1.166
_MSCN_C = 1.0


def _gaussian_window_stats(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local mean/std of `gray` under a 7x7 Gaussian(sigma=1.166) window,
    via separable Gaussian blurring of the image and the squared image
    (E[X] and E[X^2] -> Var[X] = E[X^2] - E[X]^2)."""
    import cv2

    if gray.dtype != np.float64:
        gray = gray.astype(np.float64)
    mu = cv2.GaussianBlur(gray, (_WINDOW_SIZE, _WINDOW_SIZE), _WINDOW_SIGMA, borderType=cv2.BORDER_REPLICATE)
    mu_sq = cv2.GaussianBlur(gray * gray, (_WINDOW_SIZE, _WINDOW_SIZE), _WINDOW_SIGMA, borderType=cv2.BORDER_REPLICATE)
    variance = np.maximum(mu_sq - mu * mu, 0.0)  # clip tiny negative values from float error
    sigma = np.sqrt(variance)
    return mu, sigma


def compute_mscn(gray: np.ndarray) -> np.ndarray:
    """MSCN(i,j) = (I(i,j) - mu(i,j)) / (sigma(i,j) + C)."""
    mu, sigma = _gaussian_window_stats(gray)
    return (gray.astype(np.float64) - mu) / (sigma + _MSCN_C)


def _resize_gray(gray: np.ndarray, max_side: int) -> np.ndarray:
    """Cap the longer side at `max_side` (area interpolation), mirroring
    `eyeqc.core._resize_rgb`'s own convention so every baseline in this
    manuscript operates at a comparable, standardized analysis resolution
    rather than raw, cohort-dependent native resolutions (EyePACS fundus
    photos run ~2500x3800px; running the two-scale MSCN/GGD/AGGD fit
    directly on that is both non-standard for a 2012-era NR-IQA method
    (designed around modest-resolution IQA database images) and needlessly
    slow with no accuracy benefit)."""
    h, w = gray.shape
    if max(h, w) <= max_side:
        return gray
    scale = max_side / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    import cv2

    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _fit_ggd(coefficients: np.ndarray) -> tuple[float, float]:
    """Method-of-moments fit of a zero-mean GGD to `coefficients`.

    The GGD's kurtosis-like moment ratio
        rho = E[|X|]^2 / E[X^2]
    is a monotone function of the shape parameter alpha alone (independent
    of scale), so alpha is recovered by matching the *observed* rho against
    a precomputed table of rho(alpha) and taking the closest grid point.
    sigma^2 is simply the sample variance of the coefficients.

    Returns (alpha, sigma_squared).
    """
    coefficients = coefficients.ravel()  # already float64; ravel is a view when contiguous
    variance = float(np.mean(coefficients**2))
    if variance < 1e-12:
        return 0.2, 0.0  # degenerate (flat patch): smallest grid shape, zero spread
    mean_abs = float(np.mean(np.abs(coefficients)))
    rho = variance / (mean_abs**2 + 1e-12)
    alpha = float(_GAMMA_GRID[int(np.argmin(np.abs(rho - _GGD_RATIO_TABLE)))])
    return alpha, variance


def _fit_aggd(products: np.ndarray) -> tuple[float, float, float, float]:
    """Method-of-moments fit of an Asymmetric GGD to `products` (a pairwise
    product of neighbouring MSCN coefficients).

    left_std/right_std are the sample std of the negative-side and
    positive-side values respectively. Their ratio `gamma_hat` together with
    the overall shape-moment ratio `rhat` (corrected for the left/right
    asymmetry, `rhat_norm`) is matched against the same type of lookup table
    as the symmetric GGD case, but built from the AGGD's own moment formula.
    beta_left/beta_right (the two scale parameters) follow from left_std /
    right_std once alpha is known, and the AGGD mean `eta` follows from
    beta_left, beta_right and alpha.

    Returns (alpha, eta, left_variance, right_variance).
    """
    products = products.ravel()  # already float64; ravel is a view when contiguous
    left = products[products < 0]
    right = products[products >= 0]

    left_std = float(np.sqrt(np.mean(left**2))) if left.size else 0.0
    right_std = float(np.sqrt(np.mean(right**2))) if right.size else 0.0
    if left_std < 1e-12 and right_std < 1e-12:
        return 0.2, 0.0, 0.0, 0.0

    gamma_hat = (left_std + 1e-12) / (right_std + 1e-12)
    mean_abs = float(np.mean(np.abs(products)))
    mean_sq = float(np.mean(products**2))
    rhat = (mean_abs**2 + 1e-12) / (mean_sq + 1e-12)
    rhat_norm = (rhat * (gamma_hat**3 + 1.0) * (gamma_hat + 1.0)) / ((gamma_hat**2 + 1.0) ** 2 + 1e-12)

    alpha = float(_GAMMA_GRID[int(np.argmin(np.abs(rhat_norm - _AGGD_RATIO_TABLE)))])

    gamma_ratio = gamma_fn(1.0 / alpha) / gamma_fn(3.0 / alpha)
    beta_left = left_std * np.sqrt(gamma_ratio)
    beta_right = right_std * np.sqrt(gamma_ratio)
    eta = (beta_right - beta_left) * (gamma_fn(2.0 / alpha) / gamma_fn(1.0 / alpha))

    return alpha, float(eta), float(beta_left**2), float(beta_right**2)


def _pairwise_products(mscn: np.ndarray) -> dict[str, np.ndarray]:
    """4 pairwise products of neighbouring MSCN coefficients."""
    return {
        "H": mscn[:, :-1] * mscn[:, 1:],
        "V": mscn[:-1, :] * mscn[1:, :],
        "D1": mscn[:-1, :-1] * mscn[1:, 1:],   # main diagonal
        "D2": mscn[1:, :-1] * mscn[:-1, 1:],   # secondary (anti-)diagonal
    }


def _features_one_scale(gray: np.ndarray) -> list[float]:
    """18 features (2 GGD + 4 orientations x 4 AGGD) at one resolution."""
    mscn = compute_mscn(gray)
    alpha, variance = _fit_ggd(mscn)
    features = [alpha, variance]
    products = _pairwise_products(mscn)  # compute all 4 orientations once, not once per key
    for key in ("H", "V", "D1", "D2"):
        a, eta, left_var, right_var = _fit_aggd(products[key])
        features.extend([a, eta, left_var, right_var])
    return features


FEATURE_NAMES = (
    [f"scale1_{name}" for name in ("alpha", "sigma2", *(f"{o}_{p}" for o in ("H", "V", "D1", "D2") for p in ("alpha", "eta", "varL", "varR")))]
    + [f"scale2_{name}" for name in ("alpha", "sigma2", *(f"{o}_{p}" for o in ("H", "V", "D1", "D2") for p in ("alpha", "eta", "varL", "varR")))]
)
assert len(FEATURE_NAMES) == 36


def brisque_features(image_rgb_or_gray: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Full 36-dimensional BRISQUE feature vector for one image.

    Accepts an RGB or already-grayscale array; converts to grayscale with
    OpenCV if needed, caps the longer side at `max_side` pixels (see
    `_resize_gray`), then extracts 18 features at that resolution and 18
    more after a further 2x area-interpolation downsample.
    """
    import cv2

    if image_rgb_or_gray.ndim == 3:
        gray = cv2.cvtColor(image_rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb_or_gray
    gray = gray.astype(np.float64)
    gray = _resize_gray(gray, max_side)

    scale1 = _features_one_scale(gray)

    h, w = gray.shape
    half = cv2.resize(gray, (max(w // 2, 1), max(h // 2, 1)), interpolation=cv2.INTER_AREA)
    scale2 = _features_one_scale(half)

    return np.asarray(scale1 + scale2, dtype=np.float64)
