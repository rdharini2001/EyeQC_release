"""AutoMorph (rmaphoh/AutoMorph) retinal image-quality ensemble baseline for
the EyeQC comparisons.

Zhou, Y. et al. "AutoMorph: Automated Retinal Vascular Morphology
Quantification via a Deep Learning Pipeline." Translational Vision Science &
Technology (2022). Code: https://github.com/rmaphoh/AutoMorph

What this script does
----------------------
1. Loads AutoMorph's frozen 8-model EfficientNet-b4 ensemble
   (M1_Retinal_Image_quality_EyePACS), exactly as `test_outside.py` in that
   repo does: 8x `Efficientnet_fl(pretrained=True)`, each with its own
   `best_loss_checkpoint.pth`, softmax each model's 3-way output
   (good/usable/bad) and average the 8 softmax vectors. `softmax_bad` is the
   analogue of EyeQC's `risk_bad`.
2. Applies AutoMorph's own exact preprocessing (`dataset.py`
   `BasicDataset_OUT.preprocess`): resize to 512x512, per-image z-score using
   only pixels > 0, replicate grayscale to 3 channels, transpose to CHW. We do
   NOT use `BasicDataset_OUT` itself (it only reads flat directories via
   `os.listdir`); instead a small local Dataset reads an explicit list of
   absolute image paths taken from EyeQC's own tables, with the identical
   preprocessing math copied verbatim.
3. Runs the ensemble once on EyeQC's own 12,543-image EyePACS-train
   development cohort, then calibrates two operating thresholds
   (`_choose_fail_threshold` target sensitivity 0.90, then
   `_choose_pass_threshold` target NPV 0.98) on that development population,
   using EyeQC's own threshold-selection functions imported directly from
   `eyeqc.development`. This is a threshold-*calibration* step on AutoMorph's
   already-frozen, already-trained network output -- not a refit of the
   network itself, and not an out-of-fold procedure (AutoMorph's network was
   trained once by its own authors on their own EyePACS split; there is no
   foldable decision head here the way there is for EyeQC's own logistic
   head). This is the closest honest analogue to EyeQC's own development-only
   threshold-fitting protocol that is possible for a frozen third-party
   network.
4. Applies those two fixed thresholds, unchanged, to AutoMorph's
   `softmax_bad` on EyeQC's 5 external validation cohorts (EyeQ-test, DRIMDB,
   FQS, MSHF-CFP, DeepDRiD), reading ground truth (`gt_bad`, `cluster_id`)
   from EyeQC's own `image_level_v3_{cohort}.parquet` tables -- the same
   categorical reference-label rule used for every other model in the
   manuscript. Ground truth is never recomputed here.
5. Computes AUROC/AUPRC/Brier/FAIL_sensitivity/PASS_NPV with EyeQC's own
   `eyeqc.statistics.performance` exactly as `run_validation.py` does, and
   writes a single comparison CSV to both the data_final and results/generated
   tables directories (dual-save convention used throughout this repo).

Usage:
    python run_automorph_baseline.py
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUTOMORPH_ROOT = EXTERNAL / "automorph/M1_Retinal_Image_quality_EyePACS"
CKPT_ROOT = AUTOMORPH_ROOT / "Retinal_quality" / "EyePACS_quality" / "efficientnet"
CKPT_SEEDS = ["0_seed_42", "1_seed_40", "2_seed_38", "3_seed_36",
              "4_seed_34", "5_seed_32", "6_seed_30", "7_seed_28"]

EYEQC_ROOT = REPOSITORY
CODE_DIR = REPOSITORY
DATA_FINAL = EYEQC_ROOT / "data_final"
RESULTS_DIR = REPOSITORY / "results" / "generated"

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")

OUT_DIR = Path(__file__).resolve().parent  # baselines/automorph
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]

IMG_SIZE = (512, 512)
BATCH_SIZE = 32
NUM_WORKERS = 4

# The shared venv's installed scipy (1.12.0) is ABI-incompatible with its
# installed numpy (2.4.6) -- `from scipy import ndimage` (imported by
# eyeqc.core, which eyeqc/__init__.py imports unconditionally) and
# scipy.sparse (imported transitively by sklearn, which eyeqc.development
# imports) both raise `_ARRAY_API not found`. This is a pre-existing
# environment issue, not something this script introduces, and the venv is
# shared with other concurrent jobs, so we do NOT touch the venv's installed
# packages (no pip uninstall/upgrade in place). Instead we installed a
# second, numpy2-compatible scipy (1.17.1) side-by-side into a private,
# untouched-by-anyone-else directory with
# `pip install --no-deps --target $EYEQC_EXTERNAL/automorph/_private_pylibs scipy==1.17.1`
# and prepend that directory to sys.path here, for this process only. This
# does not modify the shared venv's site-packages in any way.
_PRIVATE_LIBS = str(EXTERNAL / "automorph/_private_pylibs")
if _PRIVATE_LIBS not in sys.path:
    sys.path.insert(0, _PRIVATE_LIBS)

sys.path.insert(0, str(AUTOMORPH_ROOT))
sys.path.insert(0, str(CODE_DIR))

from model import Efficientnet_fl  # noqa: E402  (AutoMorph repo)
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import performance, cluster_bootstrap_intervals  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset: exact AutoMorph preprocessing, applied to an explicit path list
# ---------------------------------------------------------------------------
class PathListDataset(Dataset):
    """Reads images from an explicit list of absolute paths and applies
    AutoMorph's BasicDataset_OUT.preprocess logic verbatim (see dataset.py,
    lines ~108-126 of the AutoMorph repo)."""

    def __init__(self, paths: list[str], img_size=IMG_SIZE):
        self.paths = paths
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def preprocess(pil_img: Image.Image, img_size) -> np.ndarray:
        newW, newH = img_size[0], img_size[1]
        pil_img = pil_img.resize((newW, newH))
        img_array = np.array(pil_img).astype(np.float32)

        mean_value = np.mean(img_array[img_array > 0])
        std_value = np.std(img_array[img_array > 0])
        img_array = (img_array - mean_value) / std_value

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.concatenate((img_array, img_array, img_array), axis=2)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array.transpose((2, 0, 1))
        return img_array

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            image = Image.open(path).convert("RGB")
            arr = self.preprocess(image, self.img_size)
            ok = True
        except Exception:
            arr = np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.float32)
            ok = False
        return {
            "image": torch.from_numpy(arr).type(torch.FloatTensor),
            "index": index,
            "ok": ok,
        }


# ---------------------------------------------------------------------------
# Model ensemble
# ---------------------------------------------------------------------------
def load_ensemble(device: torch.device) -> list[nn.Module]:
    models = []
    for seed in CKPT_SEEDS:
        m = Efficientnet_fl(pretrained=True)
        ckpt_path = CKPT_ROOT / seed / "best_loss_checkpoint.pth"
        state = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(state)
        m.to(device=device)
        m.eval()
        models.append(m)
        print(f"  loaded {ckpt_path}", flush=True)
    return models


def _write_progress_heartbeat(tag: str, seen: int, n: int, elapsed: float) -> None:
    """Flushed-immediately progress file so an external watcher can see
    incremental state without waiting for a cohort's full pass to finish
    (which is the only point the on-disk cache parquet gets written)."""
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
    tmp.replace(final)  # atomic on same filesystem, safe for a concurrent reader


@torch.no_grad()
def run_ensemble(models: list[nn.Module], paths: list[str], device: torch.device,
                  batch_size: int = BATCH_SIZE, tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Returns (softmax_bad array aligned to `paths`, ok-mask array)."""
    dataset = PathListDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)

    n = len(paths)
    softmax_bad = np.full(n, np.nan, dtype=np.float64)
    ok_mask = np.zeros(n, dtype=bool)

    t0 = time.time()
    seen = 0
    for batch in loader:
        imgs = batch["image"].to(device=device, dtype=torch.float32)
        idx = batch["index"].numpy()
        ok = batch["ok"].numpy()

        mask_pred_all = torch.zeros((imgs.shape[0], 3), dtype=torch.float32)
        for m in models:
            pred = m(imgs)
            pred_softmax = nn.Softmax(dim=1)(pred).detach().cpu().type(torch.FloatTensor)
            mask_pred_all += pred_softmax
        mask_pred_all = (mask_pred_all / len(models)).numpy()

        softmax_bad[idx] = mask_pred_all[:, 2]
        ok_mask[idx] = ok

        seen += imgs.shape[0]
        # Print (and file-heartbeat) every batch early on so a watcher sees
        # life within seconds, then back off to every ~20 batches.
        if seen <= batch_size * 5 or seen % (batch_size * 20) == 0 or seen == n:
            elapsed = time.time() - t0
            print(f"    [{tag}] {seen}/{n} images, {elapsed:.1f}s elapsed "
                  f"({seen / max(elapsed, 1e-6):.1f} img/s)", flush=True)
            _write_progress_heartbeat(tag, seen, n, elapsed)

    # images that failed to load produce ok=False and should not be trusted
    softmax_bad[~ok_mask] = np.nan
    return softmax_bad, ok_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("\nLoading AutoMorph 8-model EfficientNet-b4 ensemble ...", flush=True)
    models = load_ensemble(device)

    # ----------------------------------------------------------------- dev
    print("\n=== Development cohort (EyeQ-train) ===", flush=True)
    dev_cache = CACHE_DIR / "dev_softmax_bad.parquet"
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["abs_path"] = train_df["image"].apply(lambda name: str(TRAIN_IMG_DIR / name))

    if dev_cache.exists():
        print(f"  loading cached development scores from {dev_cache}", flush=True)
        dev_scores = pd.read_parquet(dev_cache)
        dev_df = train_df.merge(dev_scores, on="abs_path", how="left")
    else:
        paths = train_df["abs_path"].tolist()
        softmax_bad, ok_mask = run_ensemble(models, paths, device, tag="dev")
        dev_df = train_df.copy()
        dev_df["softmax_bad"] = softmax_bad
        dev_df["ok"] = ok_mask
        dev_df[["abs_path", "softmax_bad", "ok"]].to_parquet(dev_cache, index=False)
        print(f"  cached development scores to {dev_cache}", flush=True)

    n_dev_total = len(dev_df)
    n_dev_failed = int((~dev_df["ok"]).sum()) if "ok" in dev_df else int(dev_df["softmax_bad"].isna().sum())
    dev_valid = dev_df.dropna(subset=["softmax_bad"]).copy()
    print(f"  development N={n_dev_total}, failed_to_read={n_dev_failed}, used={len(dev_valid)}", flush=True)

    y_dev = (dev_valid["quality"].astype(int) == 2).to_numpy()
    risk_dev = dev_valid["softmax_bad"].to_numpy(float)

    from sklearn.metrics import roc_auc_score
    dev_auroc = roc_auc_score(y_dev, risk_dev)
    print(f"  development-population AUROC (AutoMorph softmax_bad vs quality==Reject): {dev_auroc:.4f}", flush=True)

    fail_threshold = _choose_fail_threshold(y_dev, risk_dev, target=0.90)
    pass_threshold = _choose_pass_threshold(y_dev, risk_dev, fail_threshold, target_npv=0.98)
    print(f"  fixed thresholds -> fail_threshold={fail_threshold:.6f}  pass_threshold={pass_threshold:.6f}", flush=True)

    threshold_record = {
        "fail_threshold": fail_threshold,
        "pass_threshold": pass_threshold,
        "development_auroc": float(dev_auroc),
        "n_development_total": n_dev_total,
        "n_development_failed_to_read": n_dev_failed,
        "n_development_used": int(len(dev_valid)),
        "target_bad_sensitivity": 0.90,
        "target_pass_npv": 0.98,
    }
    (OUT_DIR / "automorph_thresholds.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- cohorts
    print("\n=== External validation cohorts ===", flush=True)
    rows = []
    drop_log = {}
    for cohort in COHORTS:
        print(f"\n[{cohort}]", flush=True)
        parquet_path = DATA_FINAL / "tables" / f"image_level_v3_{cohort}.parquet"
        coh_df = pd.read_parquet(parquet_path)

        cache_path = CACHE_DIR / f"{cohort}_softmax_bad.parquet"
        if cache_path.exists():
            print(f"  loading cached scores from {cache_path}", flush=True)
            cached = pd.read_parquet(cache_path)
            scored = coh_df.merge(cached, on="image_path", how="left")
        else:
            paths = coh_df["image_path"].astype(str).tolist()
            softmax_bad, ok_mask = run_ensemble(models, paths, device, tag=cohort)
            cached = pd.DataFrame({
                "image_path": coh_df["image_path"].astype(str).to_numpy(),
                "softmax_bad": softmax_bad,
                "ok": ok_mask,
            })
            cached.to_parquet(cache_path, index=False)
            scored = coh_df.copy()
            scored["softmax_bad"] = softmax_bad
            scored["ok"] = ok_mask

        n_total = len(scored)
        n_failed = int((~scored["ok"]).sum())
        drop_log[cohort] = {"n_total": n_total, "n_failed_to_read": n_failed}
        print(f"  N={n_total}, failed_to_read={n_failed}", flush=True)

        eval_frame = pd.DataFrame({
            "gt_bad": scored["gt_bad"],
            "risk_bad": scored["softmax_bad"],
            "cluster_id": scored["cluster_id"],
        }).dropna(subset=["gt_bad", "risk_bad"])

        point = performance(eval_frame, pass_threshold, fail_threshold)
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
        print(f"  AUROC={point['AUROC']:.3f} AUPRC={point['AUPRC']:.3f} Brier={point['Brier']:.3f} "
              f"FAIL_sens={point['FAIL_sensitivity']:.3f} PASS_NPV={point['PASS_NPV']:.3f}", flush=True)

    table = pd.DataFrame(rows)
    print("\n=== AutoMorph baseline comparison table ===", flush=True)
    print(table.round(4).to_string(index=False), flush=True)

    out_paths = [
        DATA_FINAL / "tables" / "baseline_automorph_v3.csv",
        RESULTS_DIR / "tables" / "baseline_automorph_v3.csv",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(p, index=False)
        print(f"wrote {p}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total / 60:.1f} min", flush=True)
    print(f"Thresholds: fail={fail_threshold:.6f} pass={pass_threshold:.6f}", flush=True)
    print(f"Development AUROC: {dev_auroc:.4f}", flush=True)
    print("Drop log:", json.dumps(drop_log, indent=2), flush=True)


if __name__ == "__main__":
    main()
