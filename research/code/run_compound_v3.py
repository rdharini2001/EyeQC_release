"""Compound blur x darkness interaction, v3 — same protocol as v1's
run_compound_study (100-image subsample of the degradation sample, doses
0.25/0.50/0.75/1.00, logit-scale interaction term), scored with the hybrid
model instead of the frozen ten-axis-only model."""
import json
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
from eyeqc.experiments import apply_degradation  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
MODELS = REPO_ROOT / "data_final" / "models"
DOSES = (0.25, 0.50, 0.75, 1.00)
COMPOUND_IMAGES = 100
SEED = 20260822
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main() -> None:
    contract = json.loads((MODELS / "eyeqc_v3_model.json").read_text(encoding="utf-8"))
    mean = np.asarray(contract["standardizer_mean"])
    scale = np.asarray(contract["standardizer_scale"])
    coef = np.asarray(contract["logistic_coefficients_standardized"])
    intercept = float(contract["logistic_intercept"])

    def score_one(features: dict, embedding: np.ndarray) -> float:
        x = np.concatenate([np.array([features[n] for n in FEATURES], float), embedding])
        z = (x - mean) / scale
        raw_logit = intercept + z @ coef
        calibrated_logit = contract["platt_intercept"] + contract["platt_coefficient"] * raw_logit
        return float(_sigmoid(calibrated_logit))

    rng = np.random.default_rng(SEED)
    samples = []
    for cohort in COHORTS:
        frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        good = frame[frame["gt_bad"].eq(0)]
        n = min(COMPOUND_IMAGES // len(COHORTS), len(good))
        samples.append(good.sample(n=n, random_state=int(rng.integers(1e6))))
    sample = pd.concat(samples, ignore_index=True).sample(
        n=min(COMPOUND_IMAGES, sum(len(s) for s in samples)), random_state=SEED
    )
    print(f"compound study (v3): {len(sample)} images x {len(DOSES)}x{len(DOSES)} dose grid")

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for image_index, row in enumerate(sample.itertuples()):
            base_rgb = read_rgb(row.image_path)
            base_features = extract_features(base_rgb)
            base_embedding = embed_rgb_batch([base_rgb])[0]
            baseline = score_one(base_features, base_embedding)

            blur_images, dark_images = {}, {}
            for dose in DOSES:
                blur_images[dose], _ = apply_degradation(base_rgb, "defocus", dose, np.random.default_rng(SEED + image_index))
                dark_images[dose], _ = apply_degradation(base_rgb, "darkness", dose, np.random.default_rng(SEED + image_index + 1))

            all_variants = list(blur_images.values()) + list(dark_images.values())
            all_features = list(pool.map(extract_features, all_variants))
            all_embeddings = embed_rgb_batch(all_variants)
            blur_risk = {dose: score_one(all_features[i], all_embeddings[i]) for i, dose in enumerate(DOSES)}
            dark_risk = {dose: score_one(all_features[4 + i], all_embeddings[4 + i]) for i, dose in enumerate(DOSES)}

            combined_variants, combined_meta = [], []
            for blur_dose in DOSES:
                for dark_dose in DOSES:
                    combined, _ = apply_degradation(blur_images[blur_dose], "darkness", dark_dose, np.random.default_rng(SEED + image_index + 2))
                    combined_variants.append(combined)
                    combined_meta.append((blur_dose, dark_dose))
            combined_features = list(pool.map(extract_features, combined_variants))
            combined_embeddings = embed_rgb_batch(combined_variants)

            for i, (blur_dose, dark_dose) in enumerate(combined_meta):
                combined_risk = score_one(combined_features[i], combined_embeddings[i])
                interaction = logit(combined_risk) - (logit(blur_risk[blur_dose]) + logit(dark_risk[dark_dose]) - logit(baseline))
                rows.append(
                    {
                        "image_id": row.image_id, "blur_dose": blur_dose, "dark_dose": dark_dose,
                        "baseline_risk": baseline, "blur_risk": blur_risk[blur_dose], "dark_risk": dark_risk[dark_dose],
                        "combined_risk": combined_risk, "logit_interaction": float(interaction),
                    }
                )
            if (image_index + 1) % 10 == 0:
                print(f"  {image_index + 1}/{len(sample)} images ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    results = pd.DataFrame(rows)
    results.to_parquet(TABLES / "compound_degradation_interactions_v3_raw.parquet", index=False)

    summary = results.groupby(["blur_dose", "dark_dose"]).agg(
        median_risk=("combined_risk", "median"), median_logit_interaction=("logit_interaction", "median")
    ).reset_index()
    summary.to_csv(TABLES / "compound_degradation_interactions_v3.csv", index=False)
    print("\n", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
