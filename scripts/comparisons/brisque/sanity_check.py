"""Quick correctness sanity-check for brisque_features.py before running it
at scale on 33,634 images. Checks:
  1. The 36-feature vector runs without NaN/Inf/exception on a handful of
     real development-cohort images.
  2. A sharp image and a heavily-blurred version of the SAME image produce
     visibly different MSCN statistics (blur should push the MSCN
     distribution's shape closer to Gaussian and shrink its variance --
     the textbook BRISQUE behaviour, since natural/sharp images deviate
     more from Gaussianity than blurred ones).
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brisque_features import FEATURE_NAMES, brisque_features, compute_mscn  # noqa: E402

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")


def read_rgb(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    sample = df.sample(n=8, random_state=20260822)

    print("=== 1. feature extraction on 8 random real images ===")
    any_bad = False
    for _, row in sample.iterrows():
        path = TRAIN_IMG_DIR / row["image"]
        rgb = read_rgb(path)
        feats = brisque_features(rgb)
        n_nan = int(np.isnan(feats).sum())
        n_inf = int(np.isinf(feats).sum())
        ok = n_nan == 0 and n_inf == 0 and feats.shape == (36,)
        any_bad = any_bad or not ok
        print(f"  {row['image']:>20s} quality={row['quality']}  shape={feats.shape}  nan={n_nan}  inf={n_inf}  ok={ok}")
    print(f"  ALL OK: {not any_bad}")

    print("\n=== 2. sharp vs. heavily-blurred MSCN statistics (same image) ===")
    path = TRAIN_IMG_DIR / sample.iloc[0]["image"]
    rgb = read_rgb(path)
    # Blur in uint8 space (same dtype real images arrive in) so sharp and
    # blurred go through an identical quantization pipeline -- blurring the
    # float version and truncating back to uint8 afterwards introduces its
    # own quantization-noise texture that confounds the comparison.
    rgb_blurred = cv2.GaussianBlur(rgb, (25, 25), 8.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    blurred = cv2.cvtColor(rgb_blurred, cv2.COLOR_RGB2GRAY).astype(np.float64)

    mscn_sharp = compute_mscn(gray)
    mscn_blur = compute_mscn(blurred)

    def summarize(name: str, mscn: np.ndarray) -> dict:
        flat = mscn.ravel()
        variance = float(np.var(flat))
        kurtosis = float(np.mean((flat - flat.mean()) ** 4) / (np.var(flat) ** 2 + 1e-12))
        return {"name": name, "variance": variance, "excess_kurtosis_proxy": kurtosis}

    stat_sharp = summarize("sharp", mscn_sharp)
    stat_blur = summarize("blurred", mscn_blur)
    print(f"  sharp   : variance={stat_sharp['variance']:.4f}  kurtosis_proxy={stat_sharp['excess_kurtosis_proxy']:.3f}")
    print(f"  blurred : variance={stat_blur['variance']:.4f}  kurtosis_proxy={stat_blur['excess_kurtosis_proxy']:.3f}")

    feats_sharp = brisque_features(rgb)
    feats_blur = brisque_features(rgb_blurred)
    diff = np.abs(feats_sharp - feats_blur)
    print(f"\n  mean |feature difference| sharp vs blurred: {diff.mean():.4f}")
    print(f"  max  |feature difference| sharp vs blurred: {diff.max():.4f}  ({FEATURE_NAMES[int(np.argmax(diff))]})")
    print(f"  scale1 alpha  sharp={feats_sharp[0]:.4f}  blurred={feats_blur[0]:.4f}")
    print(f"  scale1 sigma2 sharp={feats_sharp[1]:.4f}  blurred={feats_blur[1]:.4f}")

    distinguishable = diff.mean() > 0.01
    print(f"\n  SANITY CHECK PASS (features clearly differ between sharp/blurred): {distinguishable}")

    if any_bad or not distinguishable:
        raise SystemExit("SANITY CHECK FAILED -- do not proceed to the full run")
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
