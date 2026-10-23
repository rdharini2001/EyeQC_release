"""Full per-dose localization analysis for the three masked degradations
(glare, shadow, local_blur), responding to peer review: the manuscript's
main text previously reported only the single high-dose hit rates (94.3%
glare, 93.5% shadow, 38.3% local_blur). This script recomputes, at all 5
protocol doses (0, 0.25, 0.50, 0.75, 1.0), the three localization metrics
already implemented in eyeqc.experiments._localization_metrics
(localization_point_hit, localization_iou, localization_contrast) against
the apply_degradation()-returned ground-truth mask, using the same
6x6 tile_quality_map defect map the manuscript already describes as
"unchanged... identical for every EyeQC version" (it is built only from
the ten hand-crafted features, not from any decision head or embedding).

Sample: same cohort/sampling pattern as run_dose_response_v3.py (80 good-
quality images per cohort x 5 cohorts = 400 images), but only the 3
mask-bearing degradations x 5 doses = 6,000 evaluations (no embedding
needed for the hand-crafted tile grid, so this is CPU-only and much
cheaper than the full 9-mechanism dose-response study).
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from eyeqc.core import extract_features  # noqa: E402
from eyeqc.experiments import apply_degradation, tile_quality_map, _localization_metrics  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
RESULTS_TABLES = REPO_ROOT / "results_final" / "tables"

DOSES = (0.0, 0.25, 0.50, 0.75, 1.0)  # identical dose grid to run_dose_response_v3.py
MASKED_DEGRADATIONS = ("glare", "shadow", "local_blur")  # the 3 of 9 kinds with a ground-truth mask
IMAGES_PER_COHORT = 80  # same per-cohort scale as run_dose_response_v3.py (80 x 5 cohorts = 400 images)
SEED = 20260822
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
CHANCE_LEVEL = 0.20  # established manuscript reference (bottom-quintile tile selection -> 20% chance hit rate)
WORKERS = 8


def _one_image(args):
    image_path, image_id, cohort, image_index = args
    rgb = read_rgb(image_path)
    rows = []
    for kind_index, kind in enumerate(MASKED_DEGRADATIONS):
        for dose in DOSES:
            if dose == 0.0:
                row = {
                    "image_id": image_id,
                    "cohort": cohort,
                    "degradation": kind,
                    "dose": dose,
                    "localization_point_hit": np.nan,
                    "localization_iou": np.nan,
                    "localization_contrast": np.nan,
                }
                rows.append(row)
                continue
            rng = np.random.default_rng(SEED + kind_index * 10_000 + int(dose * 1000) + image_index)
            degraded, mask = apply_degradation(rgb, kind, dose, rng)
            features = extract_features(degraded)
            grid = tile_quality_map(features)
            metrics = _localization_metrics(grid, mask)
            row = {
                "image_id": image_id,
                "cohort": cohort,
                "degradation": kind,
                "dose": dose,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    samples = []
    for cohort in COHORTS:
        frame = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        good = frame[frame["gt_bad"].eq(0)]
        n = min(IMAGES_PER_COHORT, len(good))
        picked = good.sample(n=n, random_state=int(rng.integers(1e6)))
        samples.append(picked.assign(cohort_=cohort))
    sample = pd.concat(samples, ignore_index=True)
    n_evals = len(sample) * len(MASKED_DEGRADATIONS) * len(DOSES)
    print(
        f"localization (v3, full dose range): {len(sample)} images x "
        f"{len(MASKED_DEGRADATIONS)} masked mechanisms x {len(DOSES)} doses = {n_evals} evaluations"
    )

    tasks = [
        (row.image_path, str(row.image_id), str(row.cohort_), idx)
        for idx, row in enumerate(sample.itertuples())
    ]

    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i, result in enumerate(pool.map(_one_image, tasks)):
            rows.extend(result)
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(tasks)} images ({time.perf_counter() - t0:.1f}s elapsed)", flush=True)

    results = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    results.to_parquet(TABLES / "localization_full_v3_raw.parquet", index=False)

    localized = results[results["dose"] > 0].copy()
    summary = (
        localized.groupby(["degradation", "dose"])
        .agg(
            n_images=("image_id", "size"),
            mean_point_hit=("localization_point_hit", "mean"),
            median_point_hit=("localization_point_hit", "median"),
            mean_iou=("localization_iou", "mean"),
            median_iou=("localization_iou", "median"),
            mean_contrast=("localization_contrast", "mean"),
            median_contrast=("localization_contrast", "median"),
        )
        .reset_index()
    )
    summary["chance_level_point_hit"] = CHANCE_LEVEL
    summary = summary.sort_values(["degradation", "dose"]).reset_index(drop=True)

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "localization_full_v3_summary.csv", index=False)
    summary.to_csv(RESULTS_TABLES / "localization_full_v3_summary.csv", index=False)

    print("\n", summary.round(4).to_string(index=False))
    print(f"\nTotal elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
