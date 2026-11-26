"""BRISQUE (Mittal, Moorthy & Bovik, 2012) classical no-reference
image-quality baseline as a comparison method for EyeQC.

Mittal, A., Moorthy, A. K. & Bovik, A. C. "No-Reference Image Quality
Assessment in the Spatial Domain." IEEE Transactions on Image Processing
(2012).

Why a hand-rolled implementation
---------------------------------
Two off-the-shelf options were tried first and both were rejected (see
repo-wide session notes):
  - the `brisque` pip package throws `TypeError: only 0-dimensional arrays
    can be converted to Python scalars` inside its own `scale_features`
    under this venv's numpy/scipy versions -- a genuine bug in that
    package, reproducible on any input.
  - `pyiqa` pulls in a large, risky dependency tree and previously broke
    numpy/scipy ABI compatibility for the whole shared venv when installed.
Instead, `brisque_features.py` (same directory) implements BRISQUE's
36-feature extractor directly in numpy/scipy/opencv: MSCN coefficients
under a 7x7 Gaussian(sigma=1.166) window, a zero-mean GGD fit (2 params) to
the MSCN coefficients, an AGGD fit (4 params) to each of the 4 pairwise
neighbour products (H/V/D1/D2), at the native resolution and again after a
2x area-interpolation downsample -> 36 features total. See that module's
docstring for the exact method-of-moments formulas. It was sanity-checked
in `sanity_check.py` (36/36 features finite on real images; MSCN variance
drops ~10x between a sharp image and a heavily blurred copy of the same
image, confirming the pipeline responds sensibly to a real quality
degradation) before being run at scale.

Deliberate deviation from the original paper
----------------------------------------------
The original BRISQUE paper maps its 36 features to a quality score with an
SVR pretrained on the LIVE IQA database (natural photographs with human
mean-opinion-score labels for synthetic distortions like JPEG compression
and blur -- not retinal images, and not the FAIL/PASS/REVIEW triage
judgement this manuscript cares about). Using that pretrained SVR would
require external model files we do not have, and would not be a meaningful
or fair comparison for retinal fundus images regardless. Instead -- exactly
as every other baseline in this manuscript (RETFound, AutoMorph, QuickQual)
fits/calibrates its own decision head on EyeQC's own development cohort,
never an unrelated external dataset -- we fit our OWN classifier (a
standardized, L2-regularized logistic regression -- the same model family
`eyeqc.development.develop_eyeqc` uses for EyeQC's own head, chosen for
exactly that reason: it keeps the comparison apples-to-apples, is simple,
and is well-behaved on 36 continuous features at this sample size) directly
on the 36 BRISQUE features against EyeQC's own ground truth
(`quality == 2` i.e. Reject). This is both simpler than reproducing the
original SVR pipeline and more methodologically consistent with the rest of
the manuscript's comparisons.

Protocol (mirrors every other baseline this session)
-------------------------------------------------------
1. Extract 36 BRISQUE features for all 12,543 EyePACS-train development
   images (joblib-parallel, process-based, matching
   eyeqc.pipeline.extract_manifest's convention).
2. 5-fold StratifiedGroupKFold (grouped by patient id, i.e. the EyePACS
   image id before "_left"/"_right") logistic-regression fit against
   `gt_bad = (quality == 2)`, with a C grid search by mean fold AUROC
   (mirrors eyeqc.development.develop_eyeqc exactly). Out-of-fold
   predictions feed EyeQC's own `_choose_fail_threshold` (target 0.90
   sensitivity) and `_choose_pass_threshold` (target 0.98 NPV), imported
   directly from `eyeqc.development`.
3. Refit the scaler + logistic head on 100% of the development data (same
   C) for external scoring; thresholds from step 2 are carried over
   unchanged -- no recalibration on external data.
4. Extract the same 36 features for the 5 external cohorts (EyeQ-test,
   DRIMDB, FQS, MSHF-CFP, DeepDRiD), reading ground truth
   (`gt_bad`/`cluster_id`) from EyeQC's own
   `image_level_v3_{cohort}.parquet` tables -- ground truth is never
   recomputed here.
5. Score with the development-fit classifier, apply the development-fit
   thresholds unchanged, and compute AUROC/AUPRC/Brier/FAIL_sensitivity/
   PASS_NPV with `eyeqc.statistics.performance`, plus cluster-bootstrap 95%
   CIs with `eyeqc.statistics.cluster_bootstrap_intervals`, exactly as
   `run_validation.py` does.

Usage:
    python run_brisque_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).resolve().parent  # this folder
sys.path.insert(0, str(OUT_DIR))
from brisque_features import FEATURE_NAMES, brisque_features  # noqa: E402

EYEQC_ROOT = REPOSITORY
CODE_DIR = REPOSITORY
DATA_FINAL = EYEQC_ROOT / "data_final"
RESULTS_DIR = REPOSITORY / "results" / "generated"

sys.path.insert(0, str(CODE_DIR))
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")

CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
SEED = 20260822
BOOTSTRAPS = 1000
WORKERS = 8


def read_rgb(path: str) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _extract_one(path: str) -> dict:
    """One image -> 36 BRISQUE features, or an all-NaN row + error string on
    any per-image failure (matches eyeqc.pipeline._extract_one's
    ok-mask convention so one corrupt file cannot sink the whole batch)."""
    try:
        feats = brisque_features(read_rgb(path))
        return {name: float(value) for name, value in zip(FEATURE_NAMES, feats)} | {
            "extract_ok": True,
            "extract_error": None,
        }
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see eyeqc.pipeline
        return {name: float("nan") for name in FEATURE_NAMES} | {
            "extract_ok": False,
            "extract_error": f"{type(exc).__name__}: {exc}",
        }


def extract_cached(paths: list[str], cache_path: Path, tag: str) -> pd.DataFrame:
    if cache_path.exists():
        print(f"  [{tag}] loading cached features from {cache_path}", flush=True)
        cached = pd.read_parquet(cache_path)
        if list(cached["image_path"]) == list(paths):
            return cached
        print(f"  [{tag}] cache path list mismatch -- re-extracting", flush=True)

    t0 = time.time()
    rows = Parallel(n_jobs=WORKERS, prefer="processes", batch_size=16)(
        delayed(_extract_one)(path) for path in paths
    )
    elapsed = time.time() - t0
    out = pd.DataFrame(rows)
    out.insert(0, "image_path", paths)
    n_failed = int((~out["extract_ok"]).sum())
    print(f"  [{tag}] extracted {len(paths)} images in {elapsed:.1f}s "
          f"({len(paths) / max(elapsed, 1e-6):.1f} img/s), {n_failed} failed", flush=True)
    if n_failed:
        print(f"  [{tag}] first failures: {out.loc[~out['extract_ok'], ['image_path', 'extract_error']].head(5).to_dict('records')}", flush=True)
    out.to_parquet(cache_path, index=False)
    return out


def fit_logistic_head(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = SEED):
    """Mirrors eyeqc.development.develop_eyeqc's model-selection exactly:
    5-fold StratifiedGroupKFold, grid search C by mean fold AUROC, then
    out-of-fold predictions for threshold selection."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = np.logspace(-2, 2, 9)
    c_scores = []
    for c_value in c_grid:
        fold_scores = []
        for train, test in cv.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            clf = LogisticRegression(C=c_value, class_weight="balanced", solver="lbfgs",
                                      max_iter=3000, random_state=seed).fit(scaler.transform(x[train]), y[train])
            prob = clf.predict_proba(scaler.transform(x[test]))[:, 1]
            fold_scores.append(roc_auc_score(y[test], prob))
        c_scores.append(np.mean(fold_scores))
    best_c = float(c_grid[int(np.argmax(c_scores))])

    oof = np.zeros(len(x), float)
    fold_id = np.zeros(len(x), int)
    for fold, (train, test) in enumerate(cv.split(x, y, groups), start=1):
        scaler = StandardScaler().fit(x[train])
        clf = LogisticRegression(C=best_c, class_weight="balanced", solver="lbfgs",
                                  max_iter=3000, random_state=seed).fit(scaler.transform(x[train]), y[train])
        oof[test] = clf.predict_proba(scaler.transform(x[test]))[:, 1]
        fold_id[test] = fold

    return best_c, oof, fold_id, c_scores


def main() -> None:
    t_start = time.time()

    # --------------------------------------------------------------- dev set
    print("=== Development cohort (EyePACS-train) ===", flush=True)
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["abs_path"] = train_df["image"].apply(lambda name: str(TRAIN_IMG_DIR / name))
    train_df["patient_id"] = train_df["image"].str.extract(r"^(\d+)_(?:left|right)\.jpeg$")[0]
    assert train_df["patient_id"].notna().all(), "unexpected EyePACS filename pattern in dev set"

    dev_feats = extract_cached(train_df["abs_path"].tolist(), CACHE_DIR / "dev_features.parquet", "dev")
    # extract_cached preserves row order exactly matching the input path list,
    # so a positional assignment is safe and avoids a join on path strings.
    dev = train_df.copy()
    dev[FEATURE_NAMES] = dev_feats[FEATURE_NAMES].to_numpy()
    dev["extract_ok"] = dev_feats["extract_ok"].to_numpy()

    dev_valid = dev[dev["extract_ok"] & dev[FEATURE_NAMES].notna().all(axis=1)].reset_index(drop=True)
    n_dev_total, n_dev_used = len(dev), len(dev_valid)
    print(f"  development N={n_dev_total}, usable={n_dev_used}, failed={n_dev_total - n_dev_used}", flush=True)

    x_dev = dev_valid[FEATURE_NAMES].to_numpy(float)
    y_dev = (dev_valid["quality"].astype(int) == 2).to_numpy()
    groups_dev = dev_valid["patient_id"].to_numpy()

    best_c, oof, fold_id, c_scores = fit_logistic_head(x_dev, y_dev, groups_dev)
    dev_oof_auroc = roc_auc_score(y_dev, oof)
    print(f"  C grid search: best_C={best_c:.4g}  mean-fold-AUROC-per-C={[round(s, 4) for s in c_scores]}", flush=True)
    print(f"  development OUT-OF-FOLD AUROC: {dev_oof_auroc:.4f}", flush=True)

    fail_threshold = _choose_fail_threshold(y_dev, oof, target=0.90)
    pass_threshold = _choose_pass_threshold(y_dev, oof, fail_threshold, target_npv=0.98)
    print(f"  thresholds (OOF-fit): fail_threshold={fail_threshold:.6f}  pass_threshold={pass_threshold:.6f}", flush=True)

    # refit scaler + logistic head on 100% of development data for external scoring
    final_scaler = StandardScaler().fit(x_dev)
    final_clf = LogisticRegression(C=best_c, class_weight="balanced", solver="lbfgs",
                                    max_iter=3000, random_state=SEED).fit(final_scaler.transform(x_dev), y_dev)

    threshold_record = {
        "model": "BRISQUE-36feat + logistic regression (own fit on EyeQC dev cohort)",
        "best_C": best_c,
        "c_grid_mean_fold_auroc": [float(s) for s in c_scores],
        "development_oof_auroc": float(dev_oof_auroc),
        "fail_threshold": fail_threshold,
        "pass_threshold": pass_threshold,
        "n_development_total": n_dev_total,
        "n_development_used": n_dev_used,
        "n_development_failed": n_dev_total - n_dev_used,
        "target_bad_sensitivity": 0.90,
        "target_pass_npv": 0.98,
        "seed": SEED,
    }
    (OUT_DIR / "brisque_thresholds.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- cohorts
    print("\n=== External validation cohorts ===", flush=True)
    rows = []
    for cohort in COHORTS:
        print(f"\n[{cohort}]", flush=True)
        coh_df = pd.read_parquet(DATA_FINAL / "tables" / f"image_level_v3_{cohort}.parquet")
        paths = coh_df["image_path"].astype(str).tolist()

        coh_feats = extract_cached(paths, CACHE_DIR / f"{cohort}_features.parquet", cohort)
        scored = coh_df.copy()
        scored[FEATURE_NAMES] = coh_feats[FEATURE_NAMES].to_numpy()
        scored["extract_ok"] = coh_feats["extract_ok"].to_numpy()

        usable = scored["extract_ok"] & scored[FEATURE_NAMES].notna().all(axis=1)
        n_total, n_used = len(scored), int(usable.sum())
        print(f"  N={n_total}, usable={n_used}, failed={n_total - n_used}", flush=True)

        valid = scored[usable].reset_index(drop=True)
        x_ext = final_scaler.transform(valid[FEATURE_NAMES].to_numpy(float))
        risk = final_clf.predict_proba(x_ext)[:, 1]

        eval_frame = pd.DataFrame({
            "gt_bad": valid["gt_bad"].to_numpy(),
            "risk_bad": risk,
            "cluster_id": valid["cluster_id"].to_numpy(),
        })
        point = performance(eval_frame, pass_threshold, fail_threshold)
        intervals = cluster_bootstrap_intervals(
            eval_frame, pass_threshold, fail_threshold,
            bootstraps=BOOTSTRAPS, workers=4, seed=SEED + COHORTS.index(cohort) * 10_000,
        )

        row = {
            "cohort": cohort,
            "N": int(len(eval_frame)),
            "poor_N": int(eval_frame["gt_bad"].sum()),
            "AUROC": point["AUROC"],
            "AUPRC": point["AUPRC"],
            "Brier": point["Brier"],
            "FAIL_sensitivity": point["FAIL_sensitivity"],
            "PASS_NPV": point["PASS_NPV"],
        }
        for metric, (lo, hi) in intervals.items():
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
        print(f"  AUROC={point['AUROC']:.3f} AUPRC={point['AUPRC']:.3f} Brier={point['Brier']:.3f} "
              f"FAIL_sens={point['FAIL_sensitivity']:.3f} PASS_NPV={point['PASS_NPV']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    core_cols = ["cohort", "N", "poor_N", "AUROC", "AUPRC", "Brier", "FAIL_sensitivity", "PASS_NPV"]
    print("\n=== BRISQUE baseline comparison table ===", flush=True)
    print(table[core_cols].round(4).to_string(index=False), flush=True)

    out_paths = [
        DATA_FINAL / "tables" / "baseline_brisque_v3.csv",
        RESULTS_DIR / "tables" / "baseline_brisque_v3.csv",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        table[core_cols].to_csv(p, index=False)
        print(f"wrote {p}", flush=True)

    # extended table (with bootstrap CIs) kept alongside for transparency
    table.to_csv(OUT_DIR / "baseline_brisque_v3_with_CIs.csv", index=False)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total / 60:.1f} min", flush=True)
    print(f"best_C={best_c:.4g}  development OOF AUROC={dev_oof_auroc:.4f}", flush=True)
    print(f"Thresholds: fail={fail_threshold:.6f} pass={pass_threshold:.6f}", flush=True)


if __name__ == "__main__":
    main()
