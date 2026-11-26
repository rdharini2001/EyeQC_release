"""QuickQual (Engelmann, Storkey & Bernabeu, 2023, arXiv:2307.13646) retinal
image-quality baseline as a comparison method for EyeQC.

"QuickQual: Lightweight, convenient retinal image quality scoring with
off-the-shelf pretrained models." Code: https://github.com/justinengelmann/QuickQual
Reported by its authors to beat MCF-Net on the EyeQ test set
(Accuracy 0.8863 vs 0.8800, AUC 0.9687 vs 0.9588, macro one-vs-rest 3-class).

Method (verified directly from the cloned repo, README.md +
image_preprocessing.py + QuickQual_inference_example.ipynb +
compute_metrics.ipynb)
-----------------------------------------------------------------------------
1. A frozen, off-the-shelf, ImageNet-pretrained DenseNet-121
   (`timm` "densenet121.tv_in1k", confirmed identical to
   `torchvision.models.densenet121(weights="IMAGENET1K_V1")`) with its
   classifier head replaced by `Identity()` / `num_classes=0`, run in eval
   mode with no backprop, producing a 1024-dim feature vector per image.
   NOTHING about this network is fine-tuned -- it is used purely as a fixed
   feature extractor.
2. A lightweight classifier (the paper's own choice: a
   `sklearn.pipeline.Pipeline(StandardScaler(), SVC(probability=True))`,
   default hyperparameters -- confirmed by loading the authors' own shipped
   pickle below and inspecting its `repr()`, which is exactly
   `Pipeline(steps=[('standardscaler', StandardScaler()),
   ('svc', SVC(probability=True))])`, i.e. no custom C/gamma was tuned) is
   fit on top of the frozen features to predict the 3-way EyeQ quality label
   (0=Good, 1=Usable, 2=Reject). `p(Reject)` is QuickQual's bad-quality risk
   score, the analogue of EyeQC's `risk_bad`.
3. Preprocessing (image_preprocessing.py + the inference notebook): crop
   black borders (any pixel with mean RGB > 15, plus a 20px buffer), pad to
   square, then resize. The repo's own offline preprocessing script resizes
   the cropped/padded square to 1024x1024 with LANCZOS, and the inference
   example then does `F.resize(img, 512)` (on an already-square image this
   just downsizes to 512x512) followed by `to_tensor` and
   `normalize([0.5]*3, [0.5]*3)`. We fold the two resizes into one
   (crop -> pad to square -> resize directly to 512x512 with LANCZOS) for
   efficiency; this is visually and numerically equivalent to resizing
   1024->512 immediately afterwards and is not expected to materially affect
   frozen ImageNet features. The README explicitly states input images are
   "assumed to be cropped to square without large black borders" at
   "512x512 or higher" -- we replicate exactly that contract ourselves from
   EyeQC's raw images, the same way the AutoMorph baseline in this
   manuscript replicates AutoMorph's own exact preprocessing rather than
   assuming pre-cropped inputs.

IMPORTANT -- pretrained classifier head, flagged explicitly
-----------------------------------------------------------------------------
The QuickQual GitHub repo ships ALREADY-FITTED SVM pipeline
(`quickqual_dn121_512.pkl`, downloaded from the repo's GitHub Releases page)
rather than training code. `compute_metrics.ipynb` in the repo confirms this
pickle's predictions were evaluated against
`HzFu/EyeQ/master/data/Label_EyeQ_test.csv` -- i.e. the exact same official
EyeQ test split that EyeQC's own "EyeQ-test" external cohort is built from
(EyeQC's `image_level_v3_EyeQ-test.parquet` has 16,249 rows pulled straight
from EyeQ's test images/labels). The pickle was almost certainly fit on the
disjoint EyeQ *train* split (the same 12,543-image population used as
EyeQC's own development cohort here), which is standard, non-leaking
train/test practice -- BUT:
  (a) we cannot independently verify the authors never used the EyeQ test
      split for any model-selection/early decisions while developing the
      method (the repo contains no training script, only the frozen
      artifact), so we cannot rule out indirect leakage of the kind that
      affects many "beat the benchmark on the public test split" papers;
  (b) more importantly for THIS manuscript, using the authors' own
      already-fit pickle gives QuickQual an advantage no other baseline in
      this manuscript receives: every other baseline (EyeQC itself, MCF-Net,
      AutoMorph's threshold-only calibration) has its lightweight head/
      threshold fit from scratch on EyeQC's own development cohort under
      EyeQC's own honest out-of-fold protocol.
We therefore run and report BOTH variants, exactly as instructed:
  - "shipped": the authors' frozen pickle, scored directly (no refit) on
    EyeQC's development cohort purely to calibrate the two EyeQC operating
    thresholds (mirrors how the AutoMorph baseline calibrates thresholds on
    top of a frozen third-party network it does not retrain), then applied
    unchanged to the 5 external cohorts.
  - "fromscratch": our own `Pipeline(StandardScaler(), SVC(probability=True))`
    (identical architecture/hyperparameters to the shipped pickle) fit from
    scratch on EyeQC's own development cohort, with honest out-of-fold (OOF)
    predictions from a 5-fold `StratifiedGroupKFold` grouped by patient id,
    stratified on the 3-class quality label, seed 20260822 -- exactly the CV
    convention used by this manuscript's MCF-Net baseline
    (`baselines/mcfnet/train_mcfnet.py`) and EyeQC's own
    `eyeqc.development.develop_eyeqc`. This is the fair, apples-to-apples
    variant, directly comparable to every other baseline in the manuscript.

The PRIMARY output table (`baseline_quickqual_v3.csv`, the one with the
exact column schema requested) reports the "fromscratch" variant, because
that is the one methodologically comparable to every other baseline in this
manuscript. The "shipped" variant is written to a clearly-named secondary
file (`baseline_quickqual_shipped_v3.csv`, same schema, same two
directories) for transparency, and both are discussed in the run's printed
summary.

Usage:
    python run_quickqual_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision.transforms import functional as TF

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUICKQUAL_ROOT = EXTERNAL / "quickqual"
SHIPPED_PKL = QUICKQUAL_ROOT / "quickqual_dn121_512.pkl"

EYEQC_ROOT = REPOSITORY
CODE_DIR = REPOSITORY
DATA_FINAL = EYEQC_ROOT / "data_final"
RESULTS_DIR = REPOSITORY / "results" / "generated"

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")

OUT_DIR = Path(__file__).resolve().parent  # baselines/quickqual
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]

IMG_SIZE = 512
BATCH_SIZE = 48
NUM_WORKERS = 4
FEATURE_DIM = 1024  # densenet121 final feature size

SEED = 20260822
TARGET_BAD_SENSITIVITY = 0.90
TARGET_PASS_NPV = 0.98
BOOTSTRAPS = 1000

sys.path.insert(0, str(CODE_DIR))
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402


# ---------------------------------------------------------------------------
# Preprocessing (QuickQual's own: crop black borders -> pad to square ->
# resize), applied directly to an explicit list of EyeQC image paths.
# ---------------------------------------------------------------------------
def _quickqual_preprocess(pil_img: Image.Image, size: int = IMG_SIZE, threshold: int = 15, buffer: int = 20) -> Image.Image:
    img_array = np.array(pil_img)
    img_array_mean = img_array.mean(-1)
    mask = img_array_mean > threshold
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("no foreground pixels above threshold")
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())

    left = max(0, left - buffer)
    right = min(img_array.shape[1], right + buffer)
    top = max(0, top - buffer)
    bottom = min(img_array.shape[0], bottom + buffer)

    img = pil_img.crop((left, top, right, bottom))

    width, height = img.size
    if width > height:
        to_pad = width - height
        top_pad = to_pad // 2
        bottom_pad = to_pad - top_pad
        padding = [0, top_pad, 0, bottom_pad]
    else:
        to_pad = height - width
        left_pad = to_pad // 2
        right_pad = to_pad - left_pad
        padding = [left_pad, 0, right_pad, 0]
    img = TF.pad(img, padding)
    img = img.resize((size, size), resample=Image.LANCZOS)
    return img


class PathListDataset(Dataset):
    """Reads images from an explicit list of absolute paths and applies
    QuickQual's own crop/pad/resize + to_tensor + normalize(0.5, 0.5)."""

    def __init__(self, paths: list[str], size: int = IMG_SIZE):
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            image = Image.open(path).convert("RGB")
            try:
                image = _quickqual_preprocess(image, self.size)
            except Exception:
                # same fallback as the repo's own preprocess_img_mp: just resize
                image = image.resize((self.size, self.size), resample=Image.LANCZOS)
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, [0.5] * 3, [0.5] * 3)
            ok = True
        except Exception:
            tensor = torch.zeros((3, self.size, self.size), dtype=torch.float32)
            ok = False
        return {"image": tensor, "index": index, "ok": ok}


def _write_progress_heartbeat(tag: str, seen: int, n: int, elapsed: float) -> None:
    rate = seen / max(elapsed, 1e-6)
    eta_s = (n - seen) / max(rate, 1e-6)
    payload = {
        "tag": tag,
        "seen": seen,
        "total": n,
        "pct": round(100.0 * seen / max(n, 1), 2),
        "elapsed_s": round(elapsed, 1),
        "img_per_s": round(rate, 2),
        "eta_s": round(eta_s, 1),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = CACHE_DIR / "_progress.json.tmp"
    final = CACHE_DIR / "_progress.json"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(final)


@torch.no_grad()
def extract_features(model: torch.nn.Module, paths: list[str], device: torch.device,
                      batch_size: int = BATCH_SIZE, tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [n, FEATURE_DIM], ok_mask [n])."""
    dataset = PathListDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)

    n = len(paths)
    features = np.full((n, FEATURE_DIM), np.nan, dtype=np.float32)
    ok_mask = np.zeros(n, dtype=bool)

    t0 = time.time()
    seen = 0
    for batch in loader:
        imgs = batch["image"].to(device=device, dtype=torch.float32)
        idx = batch["index"].numpy()
        ok = batch["ok"].numpy()

        feats = model(imgs).detach().float().cpu().numpy()
        features[idx] = feats
        ok_mask[idx] = ok

        seen += imgs.shape[0]
        if seen <= batch_size * 5 or seen % (batch_size * 20) == 0 or seen == n:
            elapsed = time.time() - t0
            print(f"    [{tag}] {seen}/{n} images, {elapsed:.1f}s elapsed "
                  f"({seen / max(elapsed, 1e-6):.1f} img/s)", flush=True)
            _write_progress_heartbeat(tag, seen, n, elapsed)

    features[~ok_mask] = np.nan
    return features, ok_mask


def patient_group_id(image_name: str) -> str:
    """Same grouping convention as mcfnet_common.patient_group_id /
    cluster_id: '10009_left.jpeg' -> '10009' (paired eyes share a group)."""
    stem = Path(image_name).stem
    for suffix in ("_left", "_right"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def get_or_extract_features(paths: list[str], cache_name: str, model: torch.nn.Module,
                             device: torch.device, tag: str) -> tuple[np.ndarray, np.ndarray]:
    feat_cache = CACHE_DIR / f"{cache_name}_features.npy"
    ok_cache = CACHE_DIR / f"{cache_name}_ok.npy"
    if feat_cache.exists() and ok_cache.exists():
        print(f"  loading cached features from {feat_cache}", flush=True)
        features = np.load(feat_cache)
        ok_mask = np.load(ok_cache)
        if len(features) == len(paths):
            return features, ok_mask
        print("  cache length mismatch, recomputing", flush=True)
    features, ok_mask = extract_features(model, paths, device, tag=tag)
    np.save(feat_cache, features)
    np.save(ok_cache, ok_mask)
    return features, ok_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # The repo's README explicitly confirms "The timm checkpoint we used is
    # identical to the torchvision one. Thus, timm and torchvision models
    # should be interchangeable" -- we use torchvision here because this
    # shared venv does not have `huggingface_hub` installed (required by
    # timm's hub-based weight resolution) and we do not touch the shared
    # venv's installed packages for other concurrent jobs' sake (same policy
    # already documented in the AutoMorph baseline script).
    print("\nLoading frozen ImageNet-pretrained DenseNet-121 "
          "(torchvision densenet121, weights=IMAGENET1K_V1, classifier->Identity) ...", flush=True)
    backbone = tv_models.densenet121(weights="IMAGENET1K_V1")
    backbone.classifier = torch.nn.Identity()
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------- dev
    print("\n=== Development cohort (EyeQ-train) ===", flush=True)
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["abs_path"] = train_df["image"].apply(lambda name: str(TRAIN_IMG_DIR / name))
    dev_paths = train_df["abs_path"].tolist()

    dev_features, dev_ok = get_or_extract_features(dev_paths, "dev", backbone, device, tag="dev")
    n_dev_total = len(train_df)
    n_dev_failed = int((~dev_ok).sum())
    print(f"  development N={n_dev_total}, failed_to_read={n_dev_failed}", flush=True)

    dev_valid_mask = dev_ok & np.isfinite(dev_features).all(axis=1)
    dev_df = train_df[dev_valid_mask].reset_index(drop=True)
    X_dev = dev_features[dev_valid_mask]
    y_dev_multi = dev_df["quality"].astype(int).to_numpy()
    y_dev_bad = (y_dev_multi == 2).astype(int)
    groups_dev = dev_df["image"].apply(patient_group_id).to_numpy()
    print(f"  usable development images: {len(dev_df)} (patients: {len(set(groups_dev))}); "
          f"quality dist (0=Good,1=Usable,2=Reject): {np.bincount(y_dev_multi)}", flush=True)

    # --------------------------------------------------------- head: shipped
    print("\n--- Classifier head A: authors' shipped, already-fit SVM pipeline ---", flush=True)
    shipped_clf = __import__("joblib").load(SHIPPED_PKL)
    print(f"  loaded {SHIPPED_PKL}: {shipped_clf}", flush=True)

    shipped_dev_proba = shipped_clf.predict_proba(X_dev)
    shipped_dev_risk = shipped_dev_proba[:, list(shipped_clf.classes_).index(2)]
    shipped_dev_auroc = roc_auc_score(y_dev_bad, shipped_dev_risk)
    print(f"  [shipped] development-population AUROC (NOT out-of-fold -- this classifier was "
          f"already fit by the original authors, direct scoring only): {shipped_dev_auroc:.4f}", flush=True)

    shipped_fail_t = _choose_fail_threshold(y_dev_bad, shipped_dev_risk, target=TARGET_BAD_SENSITIVITY)
    shipped_pass_t = _choose_pass_threshold(y_dev_bad, shipped_dev_risk, shipped_fail_t, target_npv=TARGET_PASS_NPV)
    print(f"  [shipped] thresholds -> fail={shipped_fail_t:.6f} pass={shipped_pass_t:.6f}", flush=True)

    # ----------------------------------------------------- head: fromscratch
    print("\n--- Classifier head B: our own SVM pipeline, fit from scratch on EyeQC dev cohort ---", flush=True)

    def make_pipeline() -> Pipeline:
        return Pipeline([("standardscaler", StandardScaler()), ("svc", SVC(probability=True, random_state=SEED))])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_dev, y_dev_multi, groups_dev))
    oof_risk = np.full(len(X_dev), np.nan, dtype=np.float64)
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        t0 = time.time()
        pipe = make_pipeline()
        pipe.fit(X_dev[train_idx], y_dev_multi[train_idx])
        proba = pipe.predict_proba(X_dev[test_idx])
        class_order = list(pipe.named_steps["svc"].classes_)
        oof_risk[test_idx] = proba[:, class_order.index(2)]
        print(f"  fold {fold_i}: train={len(train_idx)} test={len(test_idx)} "
              f"({time.time() - t0:.1f}s)", flush=True)

    assert np.isfinite(oof_risk).all(), "every dev image should receive exactly one OOF prediction"
    fromscratch_dev_oof_auroc = roc_auc_score(y_dev_bad, oof_risk)
    print(f"  [fromscratch] development OOF AUROC: {fromscratch_dev_oof_auroc:.4f}", flush=True)

    fromscratch_fail_t = _choose_fail_threshold(y_dev_bad, oof_risk, target=TARGET_BAD_SENSITIVITY)
    fromscratch_pass_t = _choose_pass_threshold(y_dev_bad, oof_risk, fromscratch_fail_t, target_npv=TARGET_PASS_NPV)
    print(f"  [fromscratch] thresholds -> fail={fromscratch_fail_t:.6f} pass={fromscratch_pass_t:.6f}", flush=True)

    print("  fitting final fromscratch classifier on 100% of development data ...", flush=True)
    t0 = time.time()
    final_pipe = make_pipeline()
    final_pipe.fit(X_dev, y_dev_multi)
    final_class_order = list(final_pipe.named_steps["svc"].classes_)
    print(f"  final fit done ({time.time() - t0:.1f}s)", flush=True)

    threshold_record = {
        "shipped": {
            "fail_threshold": shipped_fail_t,
            "pass_threshold": shipped_pass_t,
            "development_auroc": float(shipped_dev_auroc),
            "development_scoring": "direct (NOT out-of-fold) -- classifier already fit by original authors",
        },
        "fromscratch": {
            "fail_threshold": fromscratch_fail_t,
            "pass_threshold": fromscratch_pass_t,
            "development_oof_auroc": float(fromscratch_dev_oof_auroc),
            "development_scoring": "5-fold StratifiedGroupKFold OOF, grouped by patient id, seed=%d" % SEED,
        },
        "n_development_total": n_dev_total,
        "n_development_failed_to_read": n_dev_failed,
        "n_development_used": int(len(dev_df)),
        "target_bad_sensitivity": TARGET_BAD_SENSITIVITY,
        "target_pass_npv": TARGET_PASS_NPV,
    }
    (OUT_DIR / "quickqual_thresholds.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- cohorts
    print("\n=== External validation cohorts ===", flush=True)
    rows_shipped, rows_fromscratch = [], []
    drop_log = {}
    for cohort in COHORTS:
        print(f"\n[{cohort}]", flush=True)
        parquet_path = DATA_FINAL / "tables" / f"image_level_v3_{cohort}.parquet"
        coh_df = pd.read_parquet(parquet_path)
        paths = coh_df["image_path"].astype(str).tolist()

        feats, ok = get_or_extract_features(paths, cohort, backbone, device, tag=cohort)
        n_total = len(coh_df)
        n_failed = int((~ok).sum())
        drop_log[cohort] = {"n_total": n_total, "n_failed_to_read": n_failed}
        print(f"  N={n_total}, failed_to_read={n_failed}", flush=True)

        valid_mask = ok & np.isfinite(feats).all(axis=1)
        X_coh = feats[valid_mask]
        coh_valid = coh_df[valid_mask].reset_index(drop=True)

        shipped_proba = shipped_clf.predict_proba(X_coh)
        shipped_class_order = list(shipped_clf.classes_)
        shipped_risk = shipped_proba[:, shipped_class_order.index(2)]

        fromscratch_proba = final_pipe.predict_proba(X_coh)
        fromscratch_risk = fromscratch_proba[:, final_class_order.index(2)]

        base_frame = pd.DataFrame({
            "gt_bad": coh_valid["gt_bad"].to_numpy(),
            "cluster_id": coh_valid["cluster_id"].to_numpy() if "cluster_id" in coh_valid else np.arange(len(coh_valid)),
        })

        for head_name, risk, fail_t, pass_t, rows in (
            ("shipped", shipped_risk, shipped_fail_t, shipped_pass_t, rows_shipped),
            ("fromscratch", fromscratch_risk, fromscratch_fail_t, fromscratch_pass_t, rows_fromscratch),
        ):
            eval_frame = base_frame.copy()
            eval_frame["risk_bad"] = risk
            eval_frame = eval_frame.dropna(subset=["gt_bad", "risk_bad"])

            point = performance(eval_frame, pass_t, fail_t)
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
            rows.append(row)
            print(f"  [{head_name}] AUROC={point['AUROC']:.3f} AUPRC={point['AUPRC']:.3f} "
                  f"Brier={point['Brier']:.3f} FAIL_sens={point['FAIL_sensitivity']:.3f} "
                  f"PASS_NPV={point['PASS_NPV']:.3f}", flush=True)

    table_fromscratch = pd.DataFrame(rows_fromscratch)
    table_shipped = pd.DataFrame(rows_shipped)

    print("\n=== QuickQual baseline comparison table (fromscratch head -- PRIMARY, apples-to-apples) ===", flush=True)
    print(table_fromscratch.round(4).to_string(index=False), flush=True)
    print("\n=== QuickQual baseline comparison table (shipped EyeQ-fit head -- SECONDARY, flagged) ===", flush=True)
    print(table_shipped.round(4).to_string(index=False), flush=True)

    out_pairs = [
        (DATA_FINAL / "tables" / "baseline_quickqual_v3.csv", table_fromscratch),
        (RESULTS_DIR / "tables" / "baseline_quickqual_v3.csv", table_fromscratch),
        (DATA_FINAL / "tables" / "baseline_quickqual_shipped_v3.csv", table_shipped),
        (RESULTS_DIR / "tables" / "baseline_quickqual_shipped_v3.csv", table_shipped),
    ]
    for p, t in out_pairs:
        p.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(p, index=False)
        print(f"wrote {p}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total / 60:.1f} min", flush=True)
    print(f"[shipped] thresholds: fail={shipped_fail_t:.6f} pass={shipped_pass_t:.6f}  "
          f"development AUROC (direct, non-OOF): {shipped_dev_auroc:.4f}", flush=True)
    print(f"[fromscratch] thresholds: fail={fromscratch_fail_t:.6f} pass={fromscratch_pass_t:.6f}  "
          f"development OOF AUROC: {fromscratch_dev_oof_auroc:.4f}", flush=True)
    print("Drop log:", json.dumps(drop_log, indent=2), flush=True)


if __name__ == "__main__":
    main()
