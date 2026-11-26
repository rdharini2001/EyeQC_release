"""HyperIQA (Su et al., CVPR 2020) general-purpose no-reference image-quality
baseline as a comparison method for EyeQC.

Su, S. et al. "Blindly Assess Image Quality in the Wild Guided by a
Self-Adaptive Hyper Network." CVPR (2020).
Code: https://github.com/SSL92/hyperIQA

This is a *general-purpose* learned NR-IQA model -- trained on KonIQ-10k
(natural photographs scraped from Flickr), with no retina-specific training
whatsoever. Poor performance on retinal fundus
images would be an expected, informative result (a negative control showing
that generic "naturalness"-based quality models do not transfer to a
domain-specific artifact-detection task), not a bug in this script.

What this script does
----------------------
1. Loads HyperIQA's frozen HyperNet (ResNet-50 backbone + hyper network that
   generates per-image TargetNet weights), using the *official* pretrained
   checkpoint released by the paper's authors on the KonIQ-10k dataset
   (`koniq_pretrained.pkl`, downloaded from the official Google Drive link in
   the SSL92/hyperIQA README -- this worked on the first try via `gdown`, no
   Baidu fallback was needed). The checkpoint loads cleanly into the
   official, unmodified `models.HyperNet` / `models.TargetNet` classes cloned
   verbatim from https://github.com/SSL92/hyperIQA into
   $EYEQC_EXTERNAL\\hyperiqa. No `pyiqa` fallback was required, so the
   shared project venv's dependency tree was never touched by this baseline
   (`pyiqa` was NOT installed anywhere for this task).
2. Applies HyperIQA's own exact test-time preprocessing from the official
   `demo.py`: `Resize((512, 384))` -> `RandomCrop(224)` -> `ToTensor()` ->
   ImageNet `Normalize`, with 10 independently-sampled random crops per
   image, each scored and averaged -- identical to the official demo's
   crop-averaging protocol, just batched across images and crops for speed
   (HyperNet's `TargetFC` uses a grouped conv1x1 that is already natively
   batched over the image dimension, so scoring B images x 10 crops in one
   forward pass is mathematically identical to the official per-image loop;
   we are not changing the model's math, only exploiting an existing batch
   dimension). DataLoader workers are seeded deterministically
   (`worker_init_fn`) so the 10 random crops are reproducible across reruns
   of this exact script with the same `NUM_WORKERS`/`BATCH_SIZE`.
3. HyperIQA outputs a 0-100 quality score where HIGHER = BETTER (opposite of
   EyeQC's `risk_bad` convention, where higher = worse). We invert with
   `risk_bad_proxy = 100 - hyperiqa_score` before any thresholding or metric
   computation, exactly as instructed.
4. Runs HyperIQA once on EyeQC's own 12,543-image EyePACS-train development
   cohort, then calibrates two operating thresholds
   (`_choose_fail_threshold` target sensitivity 0.90, then
   `_choose_pass_threshold` target NPV 0.98) on that development population,
   using EyeQC's own threshold-selection functions imported directly from
   `eyeqc.development`. This is a threshold-*calibration* step on
   HyperIQA's already-frozen, already-trained network output -- not a refit
   of the network itself, and not an out-of-fold procedure (HyperIQA's
   network was trained once by its own authors on KonIQ-10k; there is no
   foldable decision head here the way there is for EyeQC's own logistic
   head). This mirrors exactly the honest framing used for the AutoMorph
   baseline in this same manuscript
   (scripts/comparisons/automorph/run_automorph_baseline.py).
5. Applies those two fixed thresholds, unchanged, to `risk_bad_proxy` on
   EyeQC's 5 external validation cohorts (EyeQ-test, DRIMDB, FQS, MSHF-CFP,
   DeepDRiD), reading ground truth (`gt_bad`, `cluster_id`) from EyeQC's own
   `image_level_v3_{cohort}.parquet` tables -- the same categorical
   reference-label rule used for every other model in the manuscript. Ground
   truth is never recomputed here.
6. Computes AUROC/AUPRC/Brier/FAIL_sensitivity/PASS_NPV with EyeQC's own
   `eyeqc.statistics.performance`, exactly as `run_validation.py` and the
   AutoMorph baseline script do, and writes a single comparison CSV to both
   the data_final and results/generated tables directories (dual-save
   convention used throughout this repo).

Usage:
    python run_hyperiqa_baseline.py
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
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HYPERIQA_ROOT = EXTERNAL / "hyperiqa"
CKPT_PATH = HYPERIQA_ROOT / "pretrained" / "koniq_pretrained.pkl"

EYEQC_ROOT = REPOSITORY
CODE_DIR = REPOSITORY
DATA_FINAL = EYEQC_ROOT / "data_final"
RESULTS_DIR = REPOSITORY / "results" / "generated"

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")

OUT_DIR = Path(__file__).resolve().parent  # baselines/hyperiqa
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]

NUM_CROPS = 10          # official demo.py protocol: 10 random 224x224 crops, averaged
IMAGES_PER_BATCH = 32   # -> 320 crops per forward pass; reduced from 64 after a
                        # CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED crash
                        # mid-run on FQS, caused by host-RAM exhaustion from other
                        # concurrent baseline jobs sharing this machine (QuickQual,
                        # BRISQUE, a DINOv3 benchmark were all running at the same
                        # time; free system RAM was ~3.3 GB of 32 GB total at the
                        # time of the crash) -- not a bug in this script's own math.
NUM_WORKERS = 4         # reduced from 10 for the same reason: fewer concurrent
                        # DataLoader worker processes = smaller host-RAM footprint
                        # while sharing the box with other agents' jobs.
SEED = 20260822

sys.path.insert(0, str(HYPERIQA_ROOT))
sys.path.insert(0, str(CODE_DIR))

import models  # noqa: E402  (official SSL92/hyperIQA repo, unmodified)
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import performance  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset: official HyperIQA demo.py preprocessing, applied to an explicit
# list of absolute paths, with 10 random crops per image (batched).
# ---------------------------------------------------------------------------
class PathListDataset(Dataset):
    def __init__(self, paths: list[str], num_crops: int = NUM_CROPS):
        self.paths = paths
        self.num_crops = num_crops
        self.resize = torchvision.transforms.Resize((512, 384))
        self.crop = torchvision.transforms.RandomCrop(size=224)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
            img = self.resize(img)
            crops = []
            for _ in range(self.num_crops):
                c = self.crop(img)
                c = self.to_tensor(c)
                c = self.normalize(c)
                crops.append(c)
            crops = torch.stack(crops, dim=0)  # (K, 3, 224, 224)
            ok = True
        except Exception:
            crops = torch.zeros(self.num_crops, 3, 224, 224, dtype=torch.float32)
            ok = False
        return {"crops": crops, "index": index, "ok": ok}


def _worker_init_fn(worker_id: int) -> None:
    # Deterministic, reproducible random crops across reruns of this script
    # with the same NUM_WORKERS.
    torch.manual_seed(SEED + worker_id)
    np.random.seed(SEED + worker_id)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(device: torch.device) -> torch.nn.Module:
    model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)
    model_hyper.train(False)
    state = torch.load(CKPT_PATH, map_location=device)
    model_hyper.load_state_dict(state)
    for p in model_hyper.parameters():
        p.requires_grad = False
    return model_hyper


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
def run_model(model_hyper: torch.nn.Module, paths: list[str], device: torch.device,
              images_per_batch: int = IMAGES_PER_BATCH, tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Returns (hyperiqa_score array aligned to `paths`, ok-mask array).
    hyperiqa_score is the RAW 0-100 scale, higher = better quality (NOT yet
    inverted to EyeQC's risk_bad convention)."""
    torch.manual_seed(SEED)
    dataset = PathListDataset(paths)
    loader = DataLoader(
        dataset, batch_size=images_per_batch, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, drop_last=False,
        worker_init_fn=_worker_init_fn,
    )

    n = len(paths)
    hyperiqa_score = np.full(n, np.nan, dtype=np.float64)
    ok_mask = np.zeros(n, dtype=bool)

    t0 = time.time()
    seen = 0
    for batch in loader:
        crops = batch["crops"].to(device=device, dtype=torch.float32)  # (B, K, 3, 224, 224)
        idx = batch["index"].numpy()
        ok = batch["ok"].numpy()
        b, k = crops.shape[0], crops.shape[1]

        x = crops.view(b * k, 3, 224, 224)
        # Retry once on a transient cuDNN/CUDA host-allocation failure (seen
        # in practice on this shared machine when other concurrent baseline
        # jobs temporarily exhaust system RAM/VRAM): free the cache and back
        # off briefly, then retry the same batch once before giving up.
        for attempt in range(2):
            try:
                paras = model_hyper(x)
                model_target = models.TargetNet(paras)
                pred = model_target(paras["target_in_vec"]).view(b, k)
                scores = pred.mean(dim=1).detach().cpu().numpy()
                break
            except RuntimeError as exc:
                if attempt == 0 and ("CUDNN_STATUS" in str(exc) or "CUDA out of memory" in str(exc)
                                      or "HOST_ALLOCATION" in str(exc)):
                    print(f"    [{tag}] transient CUDA/cuDNN error, retrying batch after backoff: {exc}",
                          flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(5.0)
                    continue
                raise

        hyperiqa_score[idx] = scores
        ok_mask[idx] = ok

        seen += b
        if seen <= images_per_batch * 5 or seen % (images_per_batch * 20) == 0 or seen == n:
            elapsed = time.time() - t0
            print(f"    [{tag}] {seen}/{n} images, {elapsed:.1f}s elapsed "
                  f"({seen / max(elapsed, 1e-6):.1f} img/s)", flush=True)
            _write_progress_heartbeat(tag, seen, n, elapsed)

    hyperiqa_score[~ok_mask] = np.nan
    return hyperiqa_score, ok_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("\nLoading HyperIQA (official koniq_pretrained.pkl checkpoint) ...", flush=True)
    model_hyper = load_model(device)

    # ----------------------------------------------------------------- dev
    print("\n=== Development cohort (EyeQ-train) ===", flush=True)
    dev_cache = CACHE_DIR / "dev_hyperiqa_score.parquet"
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["abs_path"] = train_df["image"].apply(lambda name: str(TRAIN_IMG_DIR / name))

    if dev_cache.exists():
        print(f"  loading cached development scores from {dev_cache}", flush=True)
        dev_scores = pd.read_parquet(dev_cache)
        dev_df = train_df.merge(dev_scores, on="abs_path", how="left")
    else:
        paths = train_df["abs_path"].tolist()
        hyperiqa_score, ok_mask = run_model(model_hyper, paths, device, tag="dev")
        dev_df = train_df.copy()
        dev_df["hyperiqa_score"] = hyperiqa_score
        dev_df["ok"] = ok_mask
        dev_df[["abs_path", "hyperiqa_score", "ok"]].to_parquet(dev_cache, index=False)
        print(f"  cached development scores to {dev_cache}", flush=True)

    n_dev_total = len(dev_df)
    n_dev_failed = int((~dev_df["ok"]).sum()) if "ok" in dev_df else int(dev_df["hyperiqa_score"].isna().sum())
    dev_valid = dev_df.dropna(subset=["hyperiqa_score"]).copy()
    # Normalize to [0, 1]: eyeqc.statistics.performance's Brier term
    # (sklearn's brier_score_loss) requires y_proba in [0, 1], whereas raw
    # HyperIQA scores are on a 0-100 scale. Dividing by 100 after inverting
    # is a monotonic rescaling so it changes nothing about ranking, AUROC,
    # AUPRC, or the sensitivity/NPV properties of thresholds chosen on this
    # scale -- it only makes the Brier term well-defined.
    dev_valid["risk_bad_proxy"] = (100.0 - dev_valid["hyperiqa_score"]) / 100.0
    print(f"  development N={n_dev_total}, failed_to_read={n_dev_failed}, used={len(dev_valid)}", flush=True)

    y_dev = (dev_valid["quality"].astype(int) == 2).to_numpy()
    risk_dev = dev_valid["risk_bad_proxy"].to_numpy(float)

    from sklearn.metrics import roc_auc_score
    dev_auroc = roc_auc_score(y_dev, risk_dev)
    print(f"  development-population AUROC (100 - HyperIQA score vs quality==Reject): {dev_auroc:.4f}", flush=True)

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
        "score_convention": "risk_bad_proxy = (100 - hyperiqa_score) / 100, scaled to [0,1] "
                             "(hyperiqa_score: 0-100, higher=better)",
    }
    (OUT_DIR / "hyperiqa_thresholds.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- cohorts
    print("\n=== External validation cohorts ===", flush=True)
    rows = []
    drop_log = {}
    for cohort in COHORTS:
        print(f"\n[{cohort}]", flush=True)
        parquet_path = DATA_FINAL / "tables" / f"image_level_v3_{cohort}.parquet"
        coh_df = pd.read_parquet(parquet_path)

        cache_path = CACHE_DIR / f"{cohort}_hyperiqa_score.parquet"
        if cache_path.exists():
            print(f"  loading cached scores from {cache_path}", flush=True)
            cached = pd.read_parquet(cache_path)
            scored = coh_df.merge(cached, on="image_path", how="left")
        else:
            paths = coh_df["image_path"].astype(str).tolist()
            hyperiqa_score, ok_mask = run_model(model_hyper, paths, device, tag=cohort)
            cached = pd.DataFrame({
                "image_path": coh_df["image_path"].astype(str).to_numpy(),
                "hyperiqa_score": hyperiqa_score,
                "ok": ok_mask,
            })
            cached.to_parquet(cache_path, index=False)
            scored = coh_df.copy()
            scored["hyperiqa_score"] = hyperiqa_score
            scored["ok"] = ok_mask

        n_total = len(scored)
        n_failed = int((~scored["ok"]).sum())
        drop_log[cohort] = {"n_total": n_total, "n_failed_to_read": n_failed}
        print(f"  N={n_total}, failed_to_read={n_failed}", flush=True)

        scored["risk_bad_proxy"] = (100.0 - scored["hyperiqa_score"]) / 100.0
        eval_frame = pd.DataFrame({
            "gt_bad": scored["gt_bad"],
            "risk_bad": scored["risk_bad_proxy"],
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
    print("\n=== HyperIQA baseline comparison table ===", flush=True)
    print(table.round(4).to_string(index=False), flush=True)

    out_paths = [
        DATA_FINAL / "tables" / "baseline_hyperiqa_v3.csv",
        RESULTS_DIR / "tables" / "baseline_hyperiqa_v3.csv",
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
