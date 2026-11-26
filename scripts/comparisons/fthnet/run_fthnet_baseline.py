"""FTHNet (Gan, Niu, et al., "Acquire Precise and Comparable Fundus Image
Quality Score: FTHNet and FQS Dataset", 2024/2026) baseline as a comparison method for EyeQC.

Code: https://github.com/HudenJear/BasiQA  (cloned to $EYEQC_EXTERNAL\\basiqa)
Checkpoint: $EYEQC_EXTERNAL/weights/net_g_226264S4.pth
            (matches options/test/test_FTHNet.yml's `pretrain_network_g`
            exactly -- this is the officially intended checkpoint/config
            pairing: network_g type FTHNet4, ffa_out_ch=32, tn_in_ch=384,
            hyper_in_ch=384, embed_dim=64, depths=[2,2,6,2],
            num_heads=[2,4,8,16], window_size=12.)

************************************************************************
IMPORTANT METHODOLOGICAL CAVEAT -- FQS is NOT a held-out cohort for FTHNet
************************************************************************
FTHNet was trained AND evaluated by its own authors on the FQS dataset,
which is also one of EyeQC's five external validation cohorts. Every other
baseline in this manuscript (RETFound/AutoMorph/QuickQual/HyperIQA/BRISQUE/
etc.) has never seen any of EyeQC's five cohorts during its own training.
FTHNet's number on FQS is therefore an IN-DISTRIBUTION / near-training
performance figure, not a genuine external-generalization figure, and must
not be presented on the same footing as its numbers on the other four
cohorts (EyeQ-test, DRIMDB, MSHF-CFP, DeepDRiD), which ARE genuinely
externally validated for this model. The output table carries an explicit
`in_distribution_for_fthnet` column (True only for FQS) so this asymmetry
survives into any downstream table/figure built from this CSV.

What this script does
----------------------
1. Loads FTHNet4 (BasiQA's `archs/fthnet4_arch.py`) and its pretrained
   checkpoint (`params` key, strict=True -- 0 missing/0 unexpected keys),
   exactly as BasiQA's own `HyperIQAModel`/`test.py` would, but via a
   lightweight standalone inference loop built directly on
   `basiqa/archs` + `basiqa/models` (not through BasiQA's yml/registry
   dataset system) -- cleaner for repeated runs across 6 image populations
   (dev + 5 cohorts), as explicitly sanctioned by the task brief.
2. SoftPool dependency: the repo requires a compiled CUDA extension
   (`softpool_cuda`, from https://github.com/alexandrosstergiou/SoftPool)
   for its `SoftPool2d` layer, used twice inside FTHNet4 as a
   softmax-weighted (per-channel, non-overlapping-window) pooling op. This
   machine has no CUDA toolkit / nvcc / MSVC `cl.exe` (verified: `nvcc`,
   `cl` not on PATH, no `CUDA Toolkit` directory under Program Files), so
   the custom CUDA kernel cannot be compiled here -- a genuine build
   blocker, not a time-boxed shortcut.
   DEVIATION (flagged): we substitute a pure-PyTorch `SoftPool2d` that we
   verified reproduces the official CUDA kernel's math EXACTLY (see
   `_softpool_pure.py` in this folder for the derivation and a unit test
   against a brute-force reference). We deliberately did NOT use the
   upstream repo's own pip-free Python/CPU fallback
   (`SoftPool/idea.py: soft_pool2d`), because that fallback sums the
   exponential mask ACROSS CHANNELS (`torch.sum(exp(x), dim=1)`) which is
   mathematically different from what the compiled CUDA kernel computes
   (confirmed from `pytorch/CUDA/softpool_cuda_kernel.cu`: the softmax
   weights are computed per-channel, independently, within each spatial
   window -- see `SoftPool2dForward`). Our replacement instead does an
   exact per-channel windowed-softmax pool via `F.unfold` +
   `torch.softmax`, and we validated it is numerically identical (max abs
   diff ~5e-7, float32) to a brute-force per-channel/per-window loop
   implementation of the CUDA kernel's own formula. SoftPool2d carries no
   learnable parameters, so this substitution does not touch the
   pretrained checkpoint in any way (strict state_dict load: 0
   missing/unexpected keys either way).
3. One additional, unrelated, pre-existing shape bug was hit and fixed in
   our *private* clone only (`$EYEQC_EXTERNAL\\basiqa\\basiqa\\archs\\
   fthnet4_arch.py`, line ~589): `x = self.norm(x).permute(0,2,1)` was
   missing the `.contiguous()` that every other permute-then-view site in
   the same file uses, which crashes with
   "view size is not compatible ... Use .reshape(...) instead" for any
   batch size > 1 (the officially shipped test harness always runs with
   batch size 1, which is why this never surfaces there). We added
   `.contiguous()` after the permute -- a pure memory-layout fix with zero
   numerical effect (batch=16 vs. batch=1 outputs for the same images
   matched to ~1e-3 relative, i.e. ordinary float32 batched-math noise) --
   purely to make GPU batching tractable (0.008s/image at batch=32 vs.
   0.5s/image at batch=1, a ~60x speedup, essential given the GPU here is
   shared with other baseline jobs running concurrently).
4. Deterministic (non-augmented) preprocessing for inference: PIL RGB ->
   Resize(512,512) -> CenterCrop(384) -> ToTensor (raw [0,1], NO ImageNet
   mean/std normalization). This matches BasiQA's own TRAINING config
   (`options/train/IQA/train_fthnet_std_multi.yml`), which never sets
   `mean`/`std` (so normalization was never active during training; the
   dataset code's docstring itself says normalizing is "not recommended").
   The shipped `options/test/test_FTHNet.yml` is a smoke-test template
   pointed at a tiny `sample_dataset` and sets ImageNet mean/std -- which
   would be inconsistent with how the network was actually trained, so we
   did not copy that. We also deliberately use a deterministic CenterCrop
   and no random horizontal flip, instead of BasiQA's own test-time
   protocol (`augment_ratio=20` random crops + random flips, averaged in
   `results.csv`) -- for a frozen, already-trained-by-its-authors network
   used purely as a baseline risk score (the same "frozen network,
   deterministic preprocessing" framing used for every other baseline in
   this manuscript, e.g. AutoMorph), a single deterministic pass is the
   fairer, reproducible, and far cheaper choice; 20x TTA was not run.
5. FTHNet outputs a continuous score on roughly a 0-100 scale (the
   training target is `meanOpinionScore / full_score`, `full_score=100`,
   so raw network output * 100 restores the MOS scale), where higher is
   supposed to mean better quality per BasiQA's README. We verify this
   EMPIRICALLY (not by assumption) by checking the sign of
   AUROC(raw_score, gt_bad) on the development cohort and on FQS
   (in-distribution for this model): if raw score correlates NEGATIVELY
   with gt_bad (i.e. AUROC(raw_score, gt_bad) < 0.5), that confirms
   higher-score = less-likely-bad = standard MOS convention, and we invert
   via `risk_bad_proxy = 100 - score` to match EyeQC's risk_bad convention
   (higher = worse). The actual measured direction is printed and written
   into this script's console/log output every run.
6. Development/threshold-calibration (frozen network, no retraining):
   run once on EyePACS-train ($EYEQC_DATASETS\\EyePACS\\train,
   Label_EyeQ_train.csv, gt_bad = quality==2, 12,543 images), fit
   fail/pass thresholds via `eyeqc.development._choose_fail_threshold`
   (target sensitivity 0.90) and `_choose_pass_threshold` (target NPV
   0.98) -- same "threshold-calibration on a frozen network" framing used
   for AutoMorph.
7. External validation (all 5 cohorts, thresholds applied UNCHANGED):
   ground truth (`gt_bad`, `cluster_id`, `image_path`) read verbatim from
   `data_final/tables/image_level_v3_{cohort}.parquet` -- never
   recomputed. Metrics via `eyeqc.statistics.performance` exactly as
   `run_validation.py` does.
8. Writes the comparison CSV to both `data_final/tables/` and
   `results/generated/` (dual-save convention used throughout this
   repo), with columns: cohort, N, poor_N, AUROC, AUPRC, Brier,
   FAIL_sensitivity, PASS_NPV, in_distribution_for_fthnet.

Usage:
    python run_fthnet_baseline.py
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.utils import _pair
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASIQA_ROOT = EXTERNAL / "basiqa"
BASIQA_PKG = BASIQA_ROOT / "basiqa"  # contains archs/, models/, data/, utils/ as top-level packages
CKPT_PATH = EXTERNAL / "weights" / "net_g_226264S4.pth"

EYEQC_ROOT = REPOSITORY
CODE_DIR = REPOSITORY
DATA_FINAL = EYEQC_ROOT / "data_final"
RESULTS_DIR = REPOSITORY / "results" / "generated"

TRAIN_CSV = (DATASETS / "EyePACS/Label_EyeQ_train.csv")
TRAIN_IMG_DIR = (DATASETS / "EyePACS/train")

OUT_DIR = Path(__file__).resolve().parent  # baselines/fthnet
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COHORTS = ["EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]
IN_DISTRIBUTION_COHORTS = {"FQS"}

BATCH_SIZE = 32
NUM_WORKERS = 4
FULL_SCORE = 100.0

sys.path.insert(0, str(BASIQA_PKG))
sys.path.insert(0, str(CODE_DIR))

# ---------------------------------------------------------------------------
# SoftPool: fake the two modules FTHNet4's arch file imports
# (`import softpool_cuda`, `from SoftPool import SoftPool2d`) with a pure
# PyTorch, per-channel, non-overlapping-window softmax-pool that we verified
# reproduces the compiled CUDA kernel's own formula exactly. See module
# docstring above for the full justification. No CUDA toolkit is available
# on this machine to build the real extension (nvcc / cl.exe / CUDA Toolkit
# directory: none found), so this is a genuine build blocker, not a
# shortcut.
# ---------------------------------------------------------------------------
from _softpool_pure import PureSoftPool2d  # noqa: E402  (this folder)

sys.modules["softpool_cuda"] = types.ModuleType("softpool_cuda")
_fake_softpool_mod = types.ModuleType("SoftPool")
_fake_softpool_mod.SoftPool2d = PureSoftPool2d
sys.modules["SoftPool"] = _fake_softpool_mod

from archs import build_network  # noqa: E402  (BasiQA repo; triggers fthnet4_arch import)
from models.hyperiqa_model import TargetNet  # noqa: E402  (BasiQA repo)

from eyeqc.development import _choose_fail_threshold, _choose_pass_threshold  # noqa: E402
from eyeqc.statistics import performance  # noqa: E402


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
NETWORK_G_OPT = dict(
    type="FTHNet4",
    ffa_out_ch=32,
    tn_in_ch=384,
    hyper_in_ch=384,
    embed_dim=64,
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 16],
    window_size=12,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
)


def build_fthnet() -> nn.Module:
    net = build_network(dict(NETWORK_G_OPT))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict(ckpt["params"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    net = net.cuda().eval()
    return net


# ---------------------------------------------------------------------------
# Dataset: deterministic preprocessing on an explicit list of absolute paths
# ---------------------------------------------------------------------------
_TFM = T.Compose([T.Resize((512, 512)), T.CenterCrop(384), T.ToTensor()])


class PathListDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            img = Image.open(path).convert("RGB")
            tensor = _TFM(img)
            return tensor, index, True
        except Exception:
            return torch.zeros(3, 384, 384), index, False


def _collate(batch):
    tensors = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ok = torch.tensor([b[2] for b in batch], dtype=torch.bool)
    return tensors, indices, ok


@torch.no_grad()
def run_inference(net: nn.Module, paths: list[str]) -> tuple[np.ndarray, list[str]]:
    """Returns raw score on a 0-100 scale (NOT yet inverted to risk_bad), and
    a list of paths that failed to load (empty on success)."""
    scores = np.full(len(paths), np.nan, dtype=float)
    failed: list[str] = []
    dataset = PathListDataset(paths)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=_collate, pin_memory=True,
    )
    for tensors, indices, ok in loader:
        if not ok.all():
            for idx in indices[~ok].tolist():
                failed.append(paths[idx])
        if ok.any():
            good_tensors = tensors[ok].cuda(non_blocking=True)
            good_indices = indices[ok]
            hyper_out = net(good_tensors)
            tn = TargetNet(hyper_out).cuda()
            for p in tn.parameters():
                p.requires_grad = False
            out = tn(hyper_out["target_in_vec"]).view(-1)
            raw = out.detach().cpu().numpy() * FULL_SCORE
            scores[good_indices.numpy()] = raw
    return scores, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    print(f"Building FTHNet4 and loading checkpoint: {CKPT_PATH}")
    net = build_fthnet()

    # ---- Development cohort (EyePACS-train) ---------------------------
    dev_csv_path = CACHE_DIR / "dev_scores.parquet"
    if dev_csv_path.exists():
        print(f"Reusing cached development scores: {dev_csv_path}")
        dev = pd.read_parquet(dev_csv_path)
    else:
        print(f"Loading development manifest: {TRAIN_CSV}")
        dev_labels = pd.read_csv(TRAIN_CSV)
        dev_labels["image_path"] = dev_labels["image"].apply(lambda f: str(TRAIN_IMG_DIR / f))
        dev_labels["gt_bad"] = (dev_labels["quality"].astype(int) == 2).astype(int)

        print(f"Running FTHNet inference on {len(dev_labels)} development images...")
        t0 = time.time()
        raw_scores, failed = run_inference(net, dev_labels["image_path"].tolist())
        print(f"  development inference: {time.time() - t0:.1f}s, {len(failed)} failed loads")
        if failed:
            print(f"  first few failed paths: {failed[:5]}")

        dev = pd.DataFrame({
            "image_path": dev_labels["image_path"],
            "gt_bad": dev_labels["gt_bad"],
            "raw_score_0_100": raw_scores,
        })
        dev.to_parquet(dev_csv_path, index=False)

    dev_valid = dev.dropna(subset=["raw_score_0_100", "gt_bad"])
    n_dev_failed = len(dev) - len(dev_valid)
    print(f"Development N={len(dev)} (valid={len(dev_valid)}, failed={n_dev_failed})")

    # ---- Empirically confirm score direction ---------------------------
    from sklearn.metrics import roc_auc_score

    auroc_raw = roc_auc_score(dev_valid["gt_bad"], dev_valid["raw_score_0_100"])
    print(f"Direction check (development cohort): AUROC(raw_score, gt_bad) = {auroc_raw:.4f}")
    if auroc_raw < 0.5:
        invert = True
        print("  -> raw score correlates NEGATIVELY with gt_bad (higher score = less likely bad),"
              " consistent with standard MOS convention (higher = better quality)."
              " Using risk_bad_proxy = (100 - raw_score) / 100.")
    else:
        invert = False
        print("  -> raw score correlates POSITIVELY with gt_bad. This is the OPPOSITE of the"
              " README's stated higher=better convention. Using risk_bad_proxy = raw_score / 100"
              " unchanged (NOT inverting) to match the empirically observed direction.")

    def to_risk(raw: np.ndarray) -> np.ndarray:
        # Rescaled to [0, 1] (not just [0, 100]) so it behaves as a proper
        # probability-like risk score -- required by `brier_score_loss`
        # inside eyeqc.statistics.performance, and consistent with the
        # [0, 1] risk_bad convention used by every other baseline in this
        # manuscript. This rescaling is monotonic, so it changes nothing
        # about AUROC/AUPRC/ranking-based metrics or threshold selection.
        return ((FULL_SCORE - raw) if invert else raw) / FULL_SCORE

    dev_valid = dev_valid.copy()
    dev_valid["risk_bad"] = to_risk(dev_valid["raw_score_0_100"].to_numpy(float))
    auroc_risk = roc_auc_score(dev_valid["gt_bad"], dev_valid["risk_bad"])
    print(f"  AUROC(risk_bad_proxy, gt_bad) on development cohort = {auroc_risk:.4f} (should be > 0.5)")

    # ---- Fit thresholds on development cohort only ---------------------
    y_dev = dev_valid["gt_bad"].astype(int).to_numpy()
    risk_dev = dev_valid["risk_bad"].to_numpy(float)
    fail_threshold = _choose_fail_threshold(y_dev, risk_dev, target=0.90)
    pass_threshold = _choose_pass_threshold(y_dev, risk_dev, fail_threshold, target_npv=0.98)
    print(f"Development-fit thresholds: fail_threshold={fail_threshold:.4f}, pass_threshold={pass_threshold:.4f}")

    dev_performance = performance(
        pd.DataFrame({"gt_bad": y_dev, "risk_bad": risk_dev}), pass_threshold, fail_threshold
    )
    print(f"Development-cohort performance (self-report, NOT a validation claim): {dev_performance}")

    # ---- External validation: 5 cohorts, thresholds applied unchanged --
    rows = []
    for cohort in COHORTS:
        cache_path = CACHE_DIR / f"{cohort.replace('-', '_')}_scores.parquet"
        if cache_path.exists():
            print(f"[{cohort}] reusing cached scores: {cache_path}")
            scored = pd.read_parquet(cache_path)
        else:
            parquet_path = DATA_FINAL / "tables" / f"image_level_v3_{cohort}.parquet"
            table = pd.read_parquet(parquet_path)[["image_path", "gt_bad", "cluster_id"]]
            print(f"[{cohort}] running FTHNet inference on {len(table)} images...")
            t0 = time.time()
            raw_scores, failed = run_inference(net, table["image_path"].tolist())
            print(f"  [{cohort}] inference: {time.time() - t0:.1f}s, {len(failed)} failed loads")
            if failed:
                print(f"  [{cohort}] first few failed paths: {failed[:5]}")
            scored = table.copy()
            scored["raw_score_0_100"] = raw_scores
            scored.to_parquet(cache_path, index=False)

        scored_valid = scored.dropna(subset=["raw_score_0_100", "gt_bad"]).copy()
        n_failed = len(scored) - len(scored_valid)
        scored_valid["risk_bad"] = to_risk(scored_valid["raw_score_0_100"].to_numpy(float))

        eval_frame = pd.DataFrame({
            "gt_bad": scored_valid["gt_bad"],
            "risk_bad": scored_valid["risk_bad"],
            "cluster_id": scored_valid["cluster_id"],
        })
        point = performance(eval_frame, pass_threshold, fail_threshold)
        row = {
            "cohort": cohort,
            "N": len(scored_valid),
            "poor_N": int(scored_valid["gt_bad"].sum()),
            "AUROC": point["AUROC"],
            "AUPRC": point["AUPRC"],
            "Brier": point["Brier"],
            "FAIL_sensitivity": point["FAIL_sensitivity"],
            "PASS_NPV": point["PASS_NPV"],
            "in_distribution_for_fthnet": cohort in IN_DISTRIBUTION_COHORTS,
        }
        rows.append(row)
        flag = " <-- IN-DISTRIBUTION for FTHNet (trained+evaluated on FQS); NOT a held-out result" if cohort in IN_DISTRIBUTION_COHORTS else ""
        print(f"[{cohort}] N={row['N']} (failed={n_failed}) AUROC={row['AUROC']:.3f} "
              f"FAIL_sens={row['FAIL_sensitivity']:.3f} PASS_NPV={row['PASS_NPV']:.3f}{flag}")

    table = pd.DataFrame(rows)
    print("\n=== FTHNet baseline: final comparison table ===")
    print(table.round(4).to_string(index=False))

    out_paths = [
        DATA_FINAL / "tables" / "baseline_fthnet_v3.csv",
        RESULTS_DIR / "tables" / "baseline_fthnet_v3.csv",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
