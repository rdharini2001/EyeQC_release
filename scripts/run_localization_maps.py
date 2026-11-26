"""Localisation of simulated glare, shadow and local blur by the EyeQC maps (spatial analysis).

Sample: 80 gradable photographs per cohort (the same images, seeds and defect generator as the tile-level analysis).
The first 40 per cohort form the development half, used to choose the window size; the other 40 are held out.

Two readouts per map (exposure, contrast, detail, combined):
  tile      6 x 6 grid; is the lowest-scoring retina tile inside the defect (ties shared)?  chance = defect tile fraction
  window_K  dense maps with window side = image side / K; pixel-level AUC, point hit and IoU of (score < 0.4) against the defect

usage: python run_localization_maps.py [--workers 8] [--windows 6 8]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from eyeqc.core import extract_features  # noqa: E402
from eyeqc.experiments import apply_degradation  # noqa: E402
from eyeqc.maps import GRID, MAP_KEYS, lowest_tile_hit, quality_maps, tile_grid  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
COHORTS = ("EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD")
DEFECTS = ("glare", "shadow", "local_blur")
DOSES = (0.25, 0.50, 0.75, 1.00)
SEED = 20260822
READABLE = 0.4


def resolve(path: str) -> Path:
    """Image paths in the tables were recorded on another machine; map them into this repository."""
    text = str(path).replace("\\", "/")
    marker = "EyeQC_npj/"
    return REPO_ROOT / text[text.find(marker) + len(marker):] if marker in text else Path(text)


def sample_images(per_cohort: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    picks = []
    for cohort in COHORTS:
        table = pd.read_parquet(TABLES / f"image_level_v3_{cohort}.parquet")
        gradable = table[table.gt_bad.eq(0)]
        picks.append(gradable.sample(n=min(per_cohort, len(gradable)), random_state=int(rng.integers(1_000_000))))
    sample = pd.concat(picks, ignore_index=True)
    position = sample.groupby("cohort").cumcount()
    sample["half"] = np.where(position < per_cohort // 2, "development", "held_out")
    return sample


def defect_rng(defect: str, dose: float, index: int) -> np.random.Generator:
    return np.random.default_rng(SEED + DEFECTS.index(defect) * 10_000 + int(dose * 1000) + index)


def pixel_auc(score: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate_image(task: tuple) -> list[dict]:
    cv2.setNumThreads(1)
    index, path, image_id, cohort, half, windows = task
    rgb = read_rgb(str(resolve(path)))
    rows = []
    for defect in DEFECTS:
        for dose in DOSES:
            degraded, mask = apply_degradation(rgb, defect, dose, defect_rng(defect, dose, index))
            features = extract_features(degraded)
            base = {"image_id": image_id, "cohort": cohort, "half": half, "degradation": defect, "dose": dose}

            tiles = cv2.resize(mask.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA) >= 0.25
            for key in MAP_KEYS:
                grid = tile_grid(features, key)
                rows.append(dict(base, map=key, readout="tile", point_hit=lowest_tile_hit(grid, tiles),
                                 chance=float(tiles[np.isfinite(grid)].mean())))

            inner = features["inner_mask"]
            truth = cv2.resize(mask.astype(np.float32), inner.shape[::-1], interpolation=cv2.INTER_AREA) >= 0.5
            labels = truth[inner]
            subset = np.random.default_rng(0).choice(labels.size, min(labels.size, 20_000), replace=False)
            for window in windows:
                maps = quality_maps(features, window)
                for key in MAP_KEYS:
                    score = maps[key][inner]
                    lowest = score <= score.min() + 1e-4
                    flagged = score < READABLE
                    union = np.logical_or(flagged, labels).sum()
                    rows.append(dict(base, map=key, readout=f"window_{window}", point_hit=float(labels[lowest].mean()),
                                     chance=float(labels.mean()), auc=pixel_auc(1 - score[subset], labels[subset]),
                                     iou=float(np.logical_and(flagged, labels).sum() / union) if union else np.nan))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--windows", type=int, nargs="+", default=[6, 8])
    args = parser.parse_args()

    sample = sample_images()
    tasks = [(i, r.image_path, str(r.image_id), str(r.cohort), r.half, tuple(args.windows)) for i, r in enumerate(sample.itertuples())]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(evaluate_image, tasks, chunksize=1), start=1):
            rows.extend(result)
            if done % 50 == 0:
                print(f"{done}/{len(tasks)}", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_parquet(TABLES / "localization_maps_v3_raw.parquet", index=False)
    summary = raw.groupby(["readout", "half", "degradation", "dose", "map"], as_index=False)[["point_hit", "chance", "auc", "iou"]].mean()
    summary.to_csv(TABLES / "localization_maps_v3_summary.csv", index=False)

    chosen = raw[(raw.half == "development") & (raw.dose == 1.0) & raw.readout.str.startswith("window") & raw["map"].isin(["exposure", "detail", "quality"])]
    print("development-half mean pixel AUC by window:", chosen.groupby("readout").auc.mean().round(4).to_dict())
    print(f"images: {raw.image_id.nunique()}  rows: {len(raw)}")


if __name__ == "__main__":
    main()
