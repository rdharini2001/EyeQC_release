"""Preprocess EyeQ-train images and every external-cohort image referenced by
EyeQC's own image_level_v3_{cohort}.parquet tables through MCF-Net's own
preprocessing pipeline (HzFu/EyeQ: fundus_prep.process_without_gb + resize to
800x800), caching the result as PNGs so re-runs skip already-done files.

This script is a *driver*: it imports fundus_prep.imread / process_without_gb
/ imwrite unmodified from the cloned MCF-Net repo and does not edit that repo.
It only monkeypatches `numpy.int` (removed in numpy>=1.24) to `int` before
import, because fundus_prep.py (vendored, unmodified, written for an older
numpy) uses the now-removed alias internally.

Usage:
    python preprocess_images.py --split train
    python preprocess_images.py --split cohort --cohort EyeQ-test
    (etc. for DRIMDB, FQS, MSHF-CFP, DeepDRiD)
    python preprocess_images.py --split all   # does everything
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import numpy as np

# numpy>=1.24 removed the `np.int` alias; the vendored, unmodified fundus_prep.py
# (written for an older numpy) still uses it in two spots. Patch here, in our own
# driver, rather than editing their file.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import cv2 as cv
import pandas as pd

MCFNET_ROOT = EXTERNAL / "mcfnet/EyeQ_preprocess"
sys.path.insert(0, str(MCFNET_ROOT))
import fundus_prep as prep  # noqa: E402

EYEPACS_TRAIN_DIR = (DATASETS / "EyePACS/train")
TRAIN_LABEL_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TABLES_DIR = (REPOSITORY / "data_final/tables")
CACHE_ROOT = EXTERNAL / "mcfnet_cache"
COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(name: str) -> str:
    return _SAFE_RE.sub("_", str(name))


def _process_one(args: tuple[str, str]) -> tuple[str, str | None]:
    """Returns (dst_path, error_message_or_None)."""
    src_path, dst_path = args
    dst = Path(dst_path)
    if dst.exists():
        return dst_path, None
    try:
        img = prep.imread(src_path)
        r_img, _borders, _mask = prep.process_without_gb(img)
        r_img = cv.resize(r_img, (800, 800))
        dst.parent.mkdir(parents=True, exist_ok=True)
        prep.imwrite(str(dst), r_img)
        return dst_path, None
    except Exception as exc:  # noqa: BLE001
        return dst_path, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=2)}"


def build_train_jobs() -> list[tuple[str, str]]:
    df = pd.read_csv(TRAIN_LABEL_CSV)
    out_dir = CACHE_ROOT / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for image_name in df["image"]:
        src = EYEPACS_TRAIN_DIR / image_name
        dst = out_dir / (Path(image_name).stem + ".png")
        jobs.append((str(src), str(dst)))
    return jobs


def build_cohort_jobs(cohort: str) -> list[tuple[str, str]]:
    df = pd.read_parquet(TABLES_DIR / f"image_level_v3_{cohort}.parquet")
    out_dir = CACHE_ROOT / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for image_path, image_id in zip(df["image_path"], df["image_id"]):
        dst = out_dir / (sanitize(image_id) + ".png")
        jobs.append((str(image_path), str(dst)))
    return jobs


def run(jobs: list[tuple[str, str]], label: str, workers: int) -> list[tuple[str, str]]:
    todo = [j for j in jobs if not Path(j[1]).exists()]
    print(f"[{label}] {len(jobs)} total, {len(todo)} to process, {len(jobs) - len(todo)} cached already")
    failures: list[tuple[str, str]] = []
    if not todo:
        return failures
    with Pool(processes=workers) as pool:
        done = 0
        for dst_path, err in pool.imap_unordered(_process_one, todo, chunksize=8):
            done += 1
            if err is not None:
                failures.append((dst_path, err))
            if done % 500 == 0 or done == len(todo):
                print(f"[{label}] {done}/{len(todo)} processed, {len(failures)} failures so far")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "cohort", "all"], required=True)
    ap.add_argument("--cohort", choices=COHORTS, default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    all_failures: dict[str, list[tuple[str, str]]] = {}

    if args.split in ("train", "all"):
        jobs = build_train_jobs()
        all_failures["train"] = run(jobs, "train", args.workers)

    cohort_list = [args.cohort] if args.split == "cohort" and args.cohort else COHORTS
    if args.split in ("cohort", "all"):
        for cohort in cohort_list:
            jobs = build_cohort_jobs(cohort)
            all_failures[cohort] = run(jobs, cohort, args.workers)

    fail_log = CACHE_ROOT / "preprocess_failures.log"
    with open(fail_log, "a", encoding="utf-8") as fh:
        for label, failures in all_failures.items():
            for dst_path, err in failures:
                fh.write(f"[{label}] {dst_path}: {err}\n")

    print("\n=== summary ===")
    for label, failures in all_failures.items():
        print(f"{label}: {len(failures)} failures")
    print(f"failure log: {fail_log}")


if __name__ == "__main__":
    main()
