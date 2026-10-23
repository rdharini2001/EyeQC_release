"""Dose-response check for the hybrid v3 model, at the manuscript's exact
protocol scale (400 good-quality images, 9 mechanisms x 5 doses = 18,000
evaluations, same sampling logic as build_notebook.py's Section 9). Same
mechanism, same metric as run_dose_response_v2.py — confirms the hybrid
design restores sensitivity to defocus/overexposure/haze severity that the
DINOv2-only model lost."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_rgb_batch  # noqa: E402
from eyeqc.core import FEATURES, extract_features  # noqa: E402
from eyeqc.experiments import DEGRADATIONS, apply_degradation, degradation_summary  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

import json

TABLES = REPO_ROOT / "data_final" / "tables"
MODELS = REPO_ROOT / "data_final" / "models"
SOURCE_LABELS = REPO_ROOT / "data_final" / "source_labels"
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
IMAGES_PER_COHORT = 80  # 80 x 5 cohorts = 400 images, matching the manuscript's DEGRADATION_IMAGES protocol
SEED = 20260822
FEATURE_WORKERS = 12
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main() -> None:
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])

    def score_batch(hand_feature_rows: list[dict], embeddings: np.ndarray) -> np.ndarray:
        hand = np.array([[row[name] for name in FEATURES] for row in hand_feature_rows], float)
        x = np.concatenate([hand, embeddings], axis=1)
        z = (x - mean) / scale
        raw_logit = intercept + z @ coef
        calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
        return _sigmoid(calibrated_logit)

    rng = np.random.default_rng(SEED)
    samples = []
    for cohort in COHORTS:
        frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        good = frame[frame["gt_bad"].eq(0)]
        n = min(IMAGES_PER_COHORT, len(good))
        picked = good.sample(n=n, random_state=int(rng.integers(1e6)))
        samples.append(picked.assign(cohort_=cohort))
    sample = pd.concat(samples, ignore_index=True)
    n_evals = len(sample) * len(DEGRADATIONS) * len(DOSES)
    print(f"dose-response (v3): {len(sample)} images x {len(DEGRADATIONS)} mechanisms x {len(DOSES)} doses = {n_evals} evaluations")

    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=FEATURE_WORKERS) as pool:
        for image_index, row in enumerate(sample.itertuples()):
            base_rgb = read_rgb(row.image_path)
            for kind_index, kind in enumerate(DEGRADATIONS):
                variants = []
                for dose in DOSES:
                    if dose == 0.0:
                        img = base_rgb
                    else:
                        deg_rng = np.random.default_rng(SEED + kind_index * 10_000 + int(dose * 1000) + image_index)
                        img, _ = apply_degradation(base_rgb, kind, dose, deg_rng)
                    variants.append(img)
                feature_rows = list(pool.map(extract_features, variants))
                embeddings = embed_rgb_batch(variants)
                risk = score_batch(feature_rows, embeddings)
                pass_t, fail_t = contract["pass_risk_threshold"], contract["fail_risk_threshold"]
                for dose, r in zip(DOSES, risk):
                    verdict = "FAIL" if r >= fail_t else "PASS" if r <= pass_t else "REVIEW"
                    rows.append({"image_id": row.image_id, "cohort": row.cohort_, "degradation": kind, "dose": dose, "risk_bad": float(r), "verdict": verdict})
            if (image_index + 1) % 20 == 0:
                print(f"  {image_index + 1}/{len(sample)} images ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    results = pd.DataFrame(rows)
    results.to_parquet(TABLES / "dose_response_v3_raw.parquet", index=False)

    summary = degradation_summary(results)
    summary.to_csv(TABLES / "degradation_monotonicity_v3.csv", index=False)
    print("\n", summary.round(3).to_string(index=False))
    print(f"\nmean monotonic fraction (v3): {summary['monotonic_fraction'].mean():.3f}")

    v1_path = SOURCE_LABELS / "degradation_monotonicity.csv"
    v2_path = TABLES / "degradation_monotonicity_v2.csv"
    if v1_path.exists() and v2_path.exists():
        v1_summary = pd.read_csv(v1_path)[["degradation", "monotonic_fraction"]].rename(columns={"monotonic_fraction": "v1"})
        v2_summary = pd.read_csv(v2_path)[["degradation", "monotonic_fraction"]].rename(columns={"monotonic_fraction": "v2"})
        v3_summary = summary[["degradation", "monotonic_fraction"]].rename(columns={"monotonic_fraction": "v3"})
        merged = v1_summary.merge(v2_summary, on="degradation").merge(v3_summary, on="degradation")
        print("\nv1 vs v2 vs v3 monotonic fraction:")
        print(merged.round(3).to_string(index=False))
        merged.to_csv(TABLES / "v1_v2_v3_monotonicity.csv", index=False)


if __name__ == "__main__":
    main()
