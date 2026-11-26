"""Large-scale rerun of the dose-response check (see run_dose_response_v3.py
for the baseline 400-image / 80-per-cohort protocol this extends).

This script targets ~600 good-quality images per cohort (capped by cohort
availability: DRIMDB has only 125 good images total, MSHF-CFP only 369, so
those two cohorts use ALL their good images rather than reaching 600) instead
of 80, for a genuinely larger and more statistically powered sample.

Sample provenance: reads the SHARED sample built by
build_sample_n600percohort.py (data_final/tables/sample_v3_n600percohort.parquet),
which is a STRICT SUPERSET of the original 400-image sample: rows 0..399 are
EXACTLY the original 80/cohort x 5 cohort sample in the original order (same
SEED=20260822 draw), and rows 400+ are NEW images drawn with a separate seed
(20261103). Because the per-image degradation RNG below is seeded from
image_index (the row's position in the sample, via enumerate), the first 400
rows get IDENTICAL degraded images and therefore IDENTICAL risk scores to the
original 400-image run -- this is used as a reproducibility sanity check.
Everything else (degradation kinds, dose grid, scoring pipeline, verdict
thresholds, monotonicity metric) is unchanged from run_dose_response_v3.py.

Checkpointing: saves partial results every CHECKPOINT_EVERY images to guard
against losing a multi-hour run to a crash.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.backbone import embed_rgb_batch  # noqa: E402
from eyeqc.core import FEATURES, extract_features  # noqa: E402
from eyeqc.experiments import DEGRADATIONS, apply_degradation, degradation_summary  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

import json

TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results" / "generated"
MODELS = REPO_ROOT / "models"
SOURCE_LABELS = REPO_ROOT / "data_final" / "source_labels"
DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)
SEED = 20260822  # same base seed as run_dose_response_v3.py (per-image degradation RNG below)
FEATURE_WORKERS = 12
SAMPLE_PATH = TABLES / "sample_v3_n600percohort.parquet"
CHECKPOINT_EVERY = 200
CHECKPOINT_PATH = TABLES / "dose_response_v3_n600percohort_CHECKPOINT.parquet"


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

    sample = pd.read_parquet(SAMPLE_PATH)
    n_evals = len(sample) * len(DEGRADATIONS) * len(DOSES)
    print(f"dose-response (v3, 600/cohort target): {len(sample)} images x {len(DEGRADATIONS)} mechanisms x {len(DOSES)} doses = {n_evals} evaluations", flush=True)
    print(f"  (of which {int(sample['in_original_400'].sum())} rows are the original 400-image sample, reproduced exactly)", flush=True)

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
                    rows.append({"image_id": row.image_id, "cohort": row.cohort_, "degradation": kind, "dose": dose, "risk_bad": float(r), "verdict": verdict, "in_original_400": bool(row.in_original_400)})
            if (image_index + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                rate = elapsed / (image_index + 1)
                remaining = (len(sample) - (image_index + 1)) * rate
                print(f"  {image_index + 1}/{len(sample)} images ({elapsed:.1f}s elapsed, {rate:.2f}s/image, ~{remaining/60:.1f} min remaining)", flush=True)
            if (image_index + 1) % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_parquet(CHECKPOINT_PATH, index=False)
                print(f"  [checkpoint saved at {image_index + 1} images -> {CHECKPOINT_PATH}]", flush=True)

    results = pd.DataFrame(rows)
    results.to_parquet(TABLES / "dose_response_v3_n600percohort_raw.parquet", index=False)

    summary = degradation_summary(results)
    summary.to_csv(TABLES / "degradation_monotonicity_v3_n600percohort.csv", index=False)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_TABLES / "degradation_monotonicity_v3_n600percohort.csv", index=False)
    print("\n", summary.round(3).to_string(index=False))
    print(f"\nmean monotonic fraction (v3, n={len(sample)}): {summary['monotonic_fraction'].mean():.3f}")

    # Reproducibility sanity check: recompute monotonicity summary restricted
    # to the original-400 subset and compare with the existing v3 (400-image) file.
    orig_subset = results[results["in_original_400"]].drop(columns=["in_original_400"])
    orig_summary = degradation_summary(orig_subset)
    orig_summary.to_csv(TABLES / "degradation_monotonicity_v3_n600percohort_original400subset.csv", index=False)

    v3_400_path = TABLES / "degradation_monotonicity_v3.csv"
    if v3_400_path.exists():
        v3_400 = pd.read_csv(v3_400_path)[["degradation", "monotonic_fraction"]].rename(columns={"monotonic_fraction": "v3_400_original"})
        subset_cmp = orig_summary[["degradation", "monotonic_fraction"]].rename(columns={"monotonic_fraction": "subset_from_n600run"})
        cmp = v3_400.merge(subset_cmp, on="degradation")
        cmp["identical"] = np.isclose(cmp["v3_400_original"], cmp["subset_from_n600run"])
        print("\nReproducibility check (original-400 subset recomputed inside this larger run vs. original 400-image file):")
        print(cmp.to_string(index=False))
        cmp.to_csv(TABLES / "dose_response_reproducibility_check.csv", index=False)

    print(f"\nTotal elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
