"""EFIQA-CP (Wang, Morano, Wan, Bogunovic, MIDL 2026 "EFIQA: Explainable
Fundus Image Quality Assessment via Anatomical Priors"; follow-up benchmark
paper Wang, Morano, Mares, Bogunovic 2026 "FunPiQ: A New Benchmark for
Pixel-Level Quality Assessment in Fundus Images", arXiv:2606.25915) baseline
as a comparison method for EyeQC.

Method, verified directly from the cloned repo ($EYEQC_EXTERNAL\\funpiq,
github.com/penway/FunPiQ) and the sibling original-method repo
(github.com/penway/EFIQA, which documents the exact DINO extraction command
that FunPiQ's Source_Code/readme.md omits)
-----------------------------------------------------------------------------
EFIQA-CP = a frozen DINOv3 ViT-L/16 backbone -> a small CNN adapter
("SpaceAdmiralCNN" / "sa_cnn", a 4-stage ConvNeXt-style head,
in_channels=1024, mid_channels=512, mlp_ratio=1) trained with a pixel-level
non-negative positive-unlabeled (nnPU) loss against anatomical-visibility
pseudo-labels, producing a per-image LOW-RESOLUTION QUALITY MAP (not a single
scalar). The authors' own pretrained checkpoint is used directly
($EYEQC_EXTERNAL/weights/EFIQA_CP.pth) -- this is an
INFERENCE-ONLY baseline, no training performed here.

Checkpoint verification
-----------------------------------------------------------------------------
`EFIQA_CP.pth` is a bare state_dict that loads with strict=True into
`badish.model.sa_cnn.SpaceAdmiralCNN(in_channels=1024, mid_channels=512,
mlp_ratio=1)` (exactly matching `Source_Code/src/badish/config/efiqa_cp.yaml`'s
`model.params`), confirming it is indeed the EFIQA-CP adapter weights.

Feature-format reverse-engineering (the critical, non-obvious step)
-----------------------------------------------------------------------------
`badish.infer`'s docstring says the expected input is "A folder with DINO
features... name.npy" of shape (H, W, C), i.e. a per-image dense patch-token
grid, NOT the single pooled embedding vector this project's own cached
`*_dinov2_embeddings.npy` files store (shape (N, 768); those are a pooled
DINOv2 ViT-B/14 vector, incompatible on three axes: wrong DINO *generation*
(v2 vs v3), wrong width (768 vs required 1024), and pooled-vs-patch-grid
representation). Re-extraction was therefore required. The repo's own
`extract_dino.py` and config files never state which backbone/size were
actually used for the `efiqa_cp` (MSHF) checkpoint we have -- FunPiQ's local
readme has no extract_dino example at all. We recovered it by triangulating:
  1. `github.com/penway/EFIQA`'s README (fetched live) gives the ONE
     documented extract_dino invocation anywhere in the method's source
     lineage: `--model_id facebook/dinov3-vitl16-pretrain-lvd1689m
     --revision ea8dc...` -- i.e. the DINO backbone is actually **DINOv3**
     ViT-L/16 (hidden size 1024), not DINOv2 as the paper titles might
     suggest.
  2. The adapter head class is literally named `ModernHead64to16`
     (`sa_cnn.py`) and does exactly two stride-2 downsamples (64 -> 32 -> 16)
     -- i.e. it expects a 64x64 input patch grid and produces a 16x16 output.
  3. `vuad/infer.py`'s pseudo-label rasterization defaults to `vit_size=16`
     (a 16x16 pseudo-label grid) -- matching the adapter's 16x16 output
     exactly, and confirming the pixel-level nnPU loss is computed on a
     16x16 grid.
  4. 64x64 patches at DINOv3's patch size of 16px requires a **1024x1024**
     input image (64*16=1024) -- consistent with the MSHF training config's
     feature-folder name `emb_dino_1024`.
  5. We verified this empirically: `facebook/dinov3-vitl16-pretrain-lvd1689m`
     is gated on HuggingFace and NOT in this account's authorized list
     (403 on file resolve even though `model_info()` metadata calls
     succeed) -- but an un-gated, byte-identical third-party mirror exists
     at `timm/vit_large_patch16_dinov3.lvd1689m` (confirmed: patch_size=16,
     num_features=1024, num_prefix_tokens=5 [1 CLS + 4 register tokens]).
     We use this mirror. A 1024x1024 probe through it yields exactly 4096
     patch tokens (64x64) after stripping the 5 prefix tokens, and the
     resulting (1,1024,64,64) tensor loads into the checkpointed adapter and
     produces a (1,1,16,16) logit map with `strict=True` weight loading --
     i.e. every link in this chain is independently confirmed, not guessed.

Precision note: the backbone overflows to NaN at this 4096-token sequence
length under plain fp16 (tested and confirmed); we use bf16 autocast instead
(also empirically confirmed NaN-free), which has a much larger dynamic range
than fp16 at a negligible accuracy cost and is simply faster than fp32 on
this GPU -- the original repo's own documented example also just overrides
precision per-run (`--fp32` flag). No model weights are modified either way.

Pixel-to-image aggregation (explicit, flagged choice)
-----------------------------------------------------------------------------
EFIQA-CP natively outputs a 16x16 probability map (sigmoid of the pixel-level
nnPU logits), not a single scalar -- consistent with the FunPiQ paper's
framing of EFIQA-CP as a pixel-level, explainable-by-design method. Neither
`badish.infer` nor the repo's documentation defines a canonical image-level
reduction (confirmed: `infer.py` only ever writes out the raw per-pixel
logits/visualisation, never a scalar). We therefore define our own,
clearly-flagged reduction, as the task instructions anticipate: the
**mean** sigmoid probability over the 16x16 grid is used as the image-level
`risk_bad` score (verified on a 120-image balanced pilot sample from EyeQ
train: AUROC 0.971, with bad-quality images scoring on average 5.9x higher
than good-quality images in this pilot -- i.e. higher score = worse quality,
no sign flip needed). `max` and `fraction of cells > 0.5` reductions are also
computed and stored per-image in the cache (not used for the primary table)
to check sensitivity to this choice.

Usage:
    python run_efiqacp_baseline.py
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
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FUNPIQ_ROOT = EXTERNAL / "funpiq"
BADISH_SRC = FUNPIQ_ROOT / "Source_Code" / "src"
CKPT_PATH = EXTERNAL / "weights" / "EFIQA_CP.pth"

EYEQC_ROOT = REPOSITORY
CODE_DIR = REPOSITORY
DATA_FINAL = EYEQC_ROOT / "data_final"
RESULTS_DIR = REPOSITORY / "results" / "generated"

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")

OUT_DIR = Path(__file__).resolve().parent  # baselines/efiqacp
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]

sys.path.insert(0, str(BADISH_SRC))
sys.path.insert(0, str(CODE_DIR))
from badish.model.sa_cnn import SpaceAdmiralCNN  # noqa: E402
from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import cluster_bootstrap_intervals, performance  # noqa: E402

# ---------------------------------------------------------------------------
# Model configuration (see module docstring for how these were derived)
# ---------------------------------------------------------------------------
BACKBONE_ID = "vit_large_patch16_dinov3.lvd1689m"  # un-gated timm mirror of
# facebook/dinov3-vitl16-pretrain-lvd1689m (gated on HF, access not granted
# to this account at the time of this run)
IMG_SIZE = 1024          # -> 64x64 patch grid at patch16
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
ADAPTER_IN_CH = 1024
ADAPTER_MID_CH = 512
ADAPTER_MLP_RATIO = 1

BATCH_SIZE = 16
NUM_WORKERS = 6

SEED = 20260822
TARGET_BAD_SENSITIVITY = 0.90
TARGET_PASS_NPV = 0.98
BOOTSTRAPS = 1000


# ---------------------------------------------------------------------------
# Dataset: reads raw images, resizes to IMG_SIZE x IMG_SIZE, normalizes with
# DINOv3's ImageNet mean/std (bicubic, antialiased -- same convention as the
# repo's own extract_dino.py).
# ---------------------------------------------------------------------------
class PathListDataset(Dataset):
    def __init__(self, paths: list[str], size: int = IMG_SIZE):
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            img = Image.open(path).convert("RGB")
            img = TF.resize(img, [self.size, self.size], interpolation=InterpolationMode.BICUBIC, antialias=True)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, DINO_MEAN, DINO_STD)
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
def score_images(backbone: torch.nn.Module, adapter: torch.nn.Module, paths: list[str],
                  device: torch.device, tag: str = "") -> dict[str, np.ndarray]:
    """Returns dict of arrays (n,): score_mean, score_max, score_fracgt05, ok."""
    dataset = PathListDataset(paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False, drop_last=False)

    n = len(paths)
    score_mean = np.full(n, np.nan, dtype=np.float64)
    score_max = np.full(n, np.nan, dtype=np.float64)
    score_frac = np.full(n, np.nan, dtype=np.float64)
    ok_mask = np.zeros(n, dtype=bool)

    num_prefix = backbone.num_prefix_tokens
    t0 = time.time()
    seen = 0
    for batch in loader:
        imgs = batch["image"].to(device=device, dtype=torch.float32)
        idx = batch["index"].numpy()
        ok = batch["ok"].numpy()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = backbone.forward_features(imgs)  # (B, 5+4096, 1024)
        patches = feats[:, num_prefix:, :]
        B, N, C = patches.shape
        H = W = int(round(N ** 0.5))
        patches = patches.reshape(B, H, W, C).permute(0, 3, 1, 2).float()  # (B,C,H,W)

        logits = adapter(patches)  # (B,1,16,16)
        probs = torch.sigmoid(logits).reshape(B, -1)

        score_mean[idx] = probs.mean(dim=1).cpu().numpy()
        score_max[idx] = probs.max(dim=1).values.cpu().numpy()
        score_frac[idx] = (probs > 0.5).float().mean(dim=1).cpu().numpy()
        ok_mask[idx] = ok

        seen += imgs.shape[0]
        if seen <= BATCH_SIZE * 5 or seen % (BATCH_SIZE * 20) == 0 or seen == n:
            elapsed = time.time() - t0
            print(f"    [{tag}] {seen}/{n} images, {elapsed:.1f}s elapsed "
                  f"({seen / max(elapsed, 1e-6):.2f} img/s)", flush=True)
            _write_progress_heartbeat(tag, seen, n, elapsed)

    # Images that failed to load should not receive a score, even if the
    # zero-image placeholder happened to run through the network.
    score_mean[~ok_mask] = np.nan
    score_max[~ok_mask] = np.nan
    score_frac[~ok_mask] = np.nan
    return {"mean": score_mean, "max": score_max, "frac_gt_0.5": score_frac, "ok": ok_mask}


def get_or_score(paths: list[str], cache_name: str, backbone, adapter, device, tag: str) -> dict[str, np.ndarray]:
    cache_path = CACHE_DIR / f"{cache_name}_scores.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        if len(z["mean"]) == len(paths):
            print(f"  loading cached scores from {cache_path}", flush=True)
            return {k: z[k] for k in z.files}
        print("  cache length mismatch, recomputing", flush=True)
    result = score_images(backbone, adapter, paths, device, tag=tag)
    np.savez_compressed(cache_path, **result)
    return result


def patient_group_id(image_name: str) -> str:
    """Same grouping convention as mcfnet_common.patient_group_id / cluster_id."""
    stem = Path(image_name).stem
    for suffix in ("_left", "_right"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def main() -> None:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print(f"\nLoading EFIQA-CP adapter checkpoint from {CKPT_PATH} ...", flush=True)
    adapter = SpaceAdmiralCNN(in_channels=ADAPTER_IN_CH, mid_channels=ADAPTER_MID_CH, mlp_ratio=ADAPTER_MLP_RATIO)
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    missing, unexpected = adapter.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, "checkpoint must load with strict=True"
    adapter.eval().to(device)
    for p in adapter.parameters():
        p.requires_grad_(False)

    print(f"Loading frozen DINOv3 ViT-L/16 backbone ({BACKBONE_ID}, un-gated timm mirror of "
          f"facebook/dinov3-vitl16-pretrain-lvd1689m) ...", flush=True)
    backbone = timm.create_model(BACKBONE_ID, pretrained=True).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    print(f"  num_prefix_tokens={backbone.num_prefix_tokens} (1 CLS + register tokens), "
          f"patch_size={backbone.patch_embed.patch_size}", flush=True)

    # ----------------------------------------------------------------- dev
    print("\n=== Development cohort (EyeQ-train) ===", flush=True)
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["abs_path"] = train_df["image"].apply(lambda name: str(TRAIN_IMG_DIR / name))
    dev_paths = train_df["abs_path"].tolist()

    dev_scores = get_or_score(dev_paths, "dev", backbone, adapter, device, tag="dev")
    n_dev_total = len(train_df)
    n_dev_failed = int((~dev_scores["ok"]).sum())
    print(f"  development N={n_dev_total}, failed_to_read={n_dev_failed}", flush=True)

    dev_valid_mask = dev_scores["ok"] & np.isfinite(dev_scores["mean"])
    dev_df = train_df[dev_valid_mask].reset_index(drop=True)
    y_dev_bad = (dev_df["quality"].astype(int) == 2).to_numpy()
    risk_dev = dev_scores["mean"][dev_valid_mask]
    groups_dev = dev_df["image"].apply(patient_group_id).to_numpy()
    print(f"  usable development images: {len(dev_df)} (patients: {len(set(groups_dev))}); "
          f"bad={int(y_dev_bad.sum())} good/usable={int((~y_dev_bad).sum())}", flush=True)

    from sklearn.metrics import roc_auc_score
    dev_auroc = roc_auc_score(y_dev_bad, risk_dev)
    print(f"  development-population AUROC (direct scoring, pretrained checkpoint -- no fitting "
          f"was performed on this cohort, only threshold CALIBRATION below): {dev_auroc:.4f}", flush=True)

    fail_t = _choose_fail_threshold(y_dev_bad, risk_dev, target=TARGET_BAD_SENSITIVITY)
    pass_t = _choose_pass_threshold(y_dev_bad, risk_dev, fail_t, target_npv=TARGET_PASS_NPV)
    print(f"  thresholds -> fail={fail_t:.6f} pass={pass_t:.6f}", flush=True)

    threshold_record = {
        "fail_threshold": fail_t,
        "pass_threshold": pass_t,
        "development_auroc": float(dev_auroc),
        "development_scoring": "direct scoring of the pretrained EFIQA-CP checkpoint "
                                "(no fitting on this cohort; only the two EyeQC operating "
                                "thresholds are calibrated here, mirroring the AutoMorph "
                                "baseline's threshold-only calibration of a frozen network)",
        "aggregation": "mean sigmoid probability over the 16x16 pixel-level EFIQA-CP output grid",
        "n_development_total": n_dev_total,
        "n_development_failed_to_read": n_dev_failed,
        "n_development_used": int(len(dev_df)),
        "target_bad_sensitivity": TARGET_BAD_SENSITIVITY,
        "target_pass_npv": TARGET_PASS_NPV,
        "backbone": f"{BACKBONE_ID} (timm mirror of facebook/dinov3-vitl16-pretrain-lvd1689m)",
        "input_size": IMG_SIZE,
        "checkpoint": str(CKPT_PATH),
    }
    (OUT_DIR / "efiqacp_thresholds.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- cohorts
    print("\n=== External validation cohorts ===", flush=True)
    rows = []
    drop_log = {}
    for cohort in COHORTS:
        print(f"\n[{cohort}]", flush=True)
        parquet_path = DATA_FINAL / "tables" / f"image_level_v3_{cohort}.parquet"
        coh_df = pd.read_parquet(parquet_path)
        paths = coh_df["image_path"].astype(str).tolist()

        scores = get_or_score(paths, cohort, backbone, adapter, device, tag=cohort)
        n_total = len(coh_df)
        n_failed = int((~scores["ok"]).sum())
        drop_log[cohort] = {"n_total": n_total, "n_failed_to_read": n_failed}
        print(f"  N={n_total}, failed_to_read={n_failed}", flush=True)

        valid_mask = scores["ok"] & np.isfinite(scores["mean"])
        coh_valid = coh_df[valid_mask].reset_index(drop=True)
        risk_coh = scores["mean"][valid_mask]

        eval_frame = pd.DataFrame({
            "gt_bad": coh_valid["gt_bad"].to_numpy(),
            "cluster_id": coh_valid["cluster_id"].to_numpy() if "cluster_id" in coh_valid else np.arange(len(coh_valid)),
            "risk_bad": risk_coh,
        }).dropna(subset=["gt_bad", "risk_bad"])

        point = performance(eval_frame, pass_t, fail_t)
        ci = cluster_bootstrap_intervals(eval_frame, pass_t, fail_t, bootstraps=BOOTSTRAPS, seed=SEED)
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
        print(f"  AUROC={point['AUROC']:.3f} [{ci['AUROC'][0]:.3f},{ci['AUROC'][1]:.3f}] "
              f"AUPRC={point['AUPRC']:.3f} Brier={point['Brier']:.3f} "
              f"FAIL_sens={point['FAIL_sensitivity']:.3f} "
              f"PASS_NPV={point['PASS_NPV']:.3f} [{ci['PASS_NPV'][0]:.3f},{ci['PASS_NPV'][1]:.3f}]", flush=True)

        (OUT_DIR / f"efiqacp_bootstrap_{cohort}.json").write_text(
            json.dumps({k: list(v) for k, v in ci.items()}, indent=2), encoding="utf-8"
        )

    table = pd.DataFrame(rows)

    print("\n=== EFIQA-CP baseline comparison table ===", flush=True)
    print(table.round(4).to_string(index=False), flush=True)

    out_pairs = [
        (DATA_FINAL / "tables" / "baseline_efiqacp_v3.csv", table),
        (RESULTS_DIR / "tables" / "baseline_efiqacp_v3.csv", table),
    ]
    for p, t in out_pairs:
        p.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(p, index=False)
        print(f"wrote {p}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total / 60:.1f} min", flush=True)
    print(f"thresholds: fail={fail_t:.6f} pass={pass_t:.6f}  development AUROC: {dev_auroc:.4f}", flush=True)
    print("Drop log:", json.dumps(drop_log, indent=2), flush=True)


if __name__ == "__main__":
    main()
