"""
Task 2: determinism check for the EyeQC v3 pipeline.

Runs the SAME real image through the full pipeline TWICE and compares:
  (a) the 10 hand-crafted features (pure OpenCV/numpy -- eyeqc.core.extract_features)
  (b) the 768-dim DINOv2 embedding (eyeqc.backbone.embed_images, GPU forward pass)
  (c) the final risk_bad_v3 score + PASS/REVIEW/FAIL verdict (run_validation_v3.score())

across 10 real images spanning all 5 external cohorts.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import numpy as np
import pandas as pd

CODE_ROOT = REPOSITORY
REPO_ROOT = REPOSITORY
sys.path.insert(0, str(CODE_ROOT))

from eyeqc.core import FEATURES, extract_features  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402
from eyeqc.backbone import embed_images  # noqa: E402

MODELS = REPO_ROOT / "models"
OUT_CSV = REPO_ROOT / "data_final" / "tables" / "determinism_check_v3.csv"
OUT_CSV_COPY = REPO_ROOT / "results" / "generated" / "determinism_check_v3.csv"

IMAGES = [
    ("EyeQ-test", str(DATASETS / "EyePACS/test/1_right.jpeg")),
    ("EyeQ-test", str(DATASETS / "EyePACS/test/10001_right.jpeg")),
    ("DRIMDB", str(DATASETS / "DRIMDB/DRIMDB/Good/drimdb_good (1).jpg")),
    ("DRIMDB", str(DATASETS / "DRIMDB/DRIMDB/Good/drimdb_good (10).jpg")),
    ("FQS", str(DATASETS / "FIQSDataset/images/00000.PNG")),
    ("FQS", str(DATASETS / "FIQSDataset/images/00001.PNG")),
    ("MSHF-CFP", str(DATASETS / "MSHF/Original/CFP/Glaucoma/Glaucoma-1.jpg")),
    ("MSHF-CFP", str(DATASETS / "MSHF/Original/CFP/Glaucoma/Glaucoma-2.jpg")),
    ("DeepDRiD", str(DATASETS / "DeepDRiD/DeepDRiD-master/regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg")),
    ("DeepDRiD", str(DATASETS / "DeepDRiD/DeepDRiD-master/regular_fundus_images/regular-fundus-training/Images/1/1_l2.jpg")),
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def score(x: np.ndarray, contract: dict) -> np.ndarray:
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])
    z = (x - mean) / scale
    raw_logit = intercept + z @ coef
    calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
    return _sigmoid(calibrated_logit)


def verdict_of(risk: float, pass_t: float, fail_t: float) -> str:
    if risk >= fail_t:
        return "FAIL"
    if risk <= pass_t:
        return "PASS"
    return "REVIEW"


def main():
    v3_contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    pass_t = float(v3_contract["pass_risk_threshold"])
    fail_t = float(v3_contract["fail_risk_threshold"])

    rows = []
    for cohort, path in IMAGES:
        p = Path(path)
        assert p.exists(), f"missing {path}"

        # ---- (a) hand-crafted features, run twice ----
        rgb = read_rgb(p)
        feats_run1 = extract_features(rgb)
        rgb2 = read_rgb(p)  # re-read + re-decode from disk independently
        feats_run2 = extract_features(rgb2)
        fvec1 = np.array([feats_run1[k] for k in FEATURES], dtype=float)
        fvec2 = np.array([feats_run2[k] for k in FEATURES], dtype=float)
        feat_identical = bool(np.array_equal(fvec1, fvec2))
        feat_max_abs_diff = float(np.max(np.abs(fvec1 - fvec2))) if not feat_identical else 0.0

        # ---- (b) DINOv2 embedding, run twice (separate embed_images calls -> fresh forward pass) ----
        emb1, ok1 = embed_images([str(p)], batch_size=1, verbose=False)
        emb2, ok2 = embed_images([str(p)], batch_size=1, verbose=False)
        assert ok1[0] and ok2[0]
        emb1, emb2 = emb1[0], emb2[0]
        emb_identical = bool(np.array_equal(emb1, emb2))
        emb_max_abs_diff = float(np.max(np.abs(emb1.astype(np.float64) - emb2.astype(np.float64))))
        emb_rel_diff = float(emb_max_abs_diff / (np.max(np.abs(emb1)) + 1e-12))

        # ---- (c) final risk score + verdict, both combinations ----
        x1 = np.concatenate([fvec1, emb1.astype(float)])[None, :]
        x2 = np.concatenate([fvec2, emb2.astype(float)])[None, :]
        risk1 = float(score(x1, v3_contract)[0])
        risk2 = float(score(x2, v3_contract)[0])
        verdict1 = verdict_of(risk1, pass_t, fail_t)
        verdict2 = verdict_of(risk2, pass_t, fail_t)

        rows.append({
            "cohort": cohort,
            "image_path": str(p),
            "hand_features_bit_identical": feat_identical,
            "hand_features_max_abs_diff": feat_max_abs_diff,
            "dinov2_embedding_bit_identical": emb_identical,
            "dinov2_embedding_max_abs_diff": emb_max_abs_diff,
            "dinov2_embedding_max_rel_diff": emb_rel_diff,
            "risk_bad_v3_run1": risk1,
            "risk_bad_v3_run2": risk2,
            "risk_bad_v3_abs_diff": abs(risk1 - risk2),
            "verdict_run1": verdict1,
            "verdict_run2": verdict2,
            "verdict_changed": verdict1 != verdict2,
        })
        print(f"[{cohort}] {p.name}: feats_identical={feat_identical} emb_identical={emb_identical} "
              f"emb_max_abs_diff={emb_max_abs_diff:.3e} risk1={risk1:.6f} risk2={risk2:.6f} "
              f"verdict={verdict1}/{verdict2}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_CSV_COPY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV_COPY, index=False)

    print("\n=== SUMMARY ===")
    print(f"hand features bit-identical: {df['hand_features_bit_identical'].sum()}/{len(df)}")
    print(f"DINOv2 embedding bit-identical: {df['dinov2_embedding_bit_identical'].sum()}/{len(df)}")
    print(f"max embedding abs diff across all images: {df['dinov2_embedding_max_abs_diff'].max():.3e}")
    print(f"max risk_bad_v3 abs diff across all images: {df['risk_bad_v3_abs_diff'].max():.3e}")
    print(f"verdict changed on any run: {df['verdict_changed'].sum()}/{len(df)}")
    print(f"\nSaved {OUT_CSV}\nand {OUT_CSV_COPY}")


if __name__ == "__main__":
    main()
