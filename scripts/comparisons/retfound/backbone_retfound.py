"""Frozen RETFound visual backbone for the EyeQC v3 ablation.

Companion to ``backbone.py``: that module's docstring notes that RETFound (a
fundus-specific self-supervised ViT) would likely be a better-matched
backbone than general-purpose DINOv2, but its weights were gated behind a
HuggingFace access request that was not available at build time. Access is
now available, so this module drops RETFound in as an alternative frozen
feature extractor without touching ``backbone.py`` or anything downstream of
it -- exactly as that docstring anticipated.

Checkpoint: ``YukunZhou/RETFound_mae_natureCFP`` (Zhou et al., Nature 2023,
"A foundation model for generalizable disease detection from retinal
images"), a ViT-Large/16 MAE encoder self-supervised on 1.6M colour fundus
photographs. Loaded with the author-sanctioned pattern from the RETFound
repo's own ``latent_feature.ipynb`` (cloned at
``$EYEQC_EXTERNAL\\retfound``): build ``models_vit.RETFound_mae(img_size
=224, global_pool=True)`` and load ``checkpoint["model"]`` with
``strict=False`` (the notebook builds the model with a throwaway
``num_classes=5`` head for a different downstream task; that head is never
used here -- only ``forward_features``, i.e. the frozen encoder, is called).

Preprocessing also follows that notebook exactly: resize to 224x224, scale
to [0, 1], then standardize *per image, per channel* using that image's own
channel mean/std (not ImageNet mean/std, and not a dataset-level statistic).
This is unusual relative to DINOv2's preprocessing but is how RETFound's own
authors extract linear-probe features from this checkpoint, so it is
followed verbatim rather than substituted for the more common ImageNet
normalization.

Kept as a pure frozen feature extractor: no fine-tuning, ``requires_grad_
(False)`` on every parameter, ``.eval()`` mode. Embedding dimension is
confirmed empirically in this module's self-test rather than assumed (see
the ``if __name__ == "__main__"`` block) and is expected to be 1024 for
ViT-Large's globally-pooled, LayerNorm'd token.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods

import cv2
import numpy as np
import torch

CODE_ROOT = REPOSITORY
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eyeqc.pipeline import read_rgb  # noqa: E402  (path must be set up first)

# The RETFound authors' own repo, cloned locally -- this gives us
# models_vit.py, the exact architecture this checkpoint was trained with.
RETFOUND_REPO_DIR = EXTERNAL / "retfound"
if str(RETFOUND_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(RETFOUND_REPO_DIR))

import models_vit as _retfound_models  # noqa: E402  (path must be set up first)

RETFOUND_HF_REPO_ID = "YukunZhou/RETFound_mae_natureCFP"
RETFOUND_HF_FILENAME = "RETFound_mae_natureCFP.pth"
INPUT_SIZE = 224
EMBED_DIM = 1024  # ViT-Large global-pooled token; confirmed empirically, see module self-test
BACKBONE_NAME = "retfound_mae_vitl16_natureCFP"

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def load_backbone(device: str | None = None) -> tuple[torch.nn.Module, str]:
    """Load the frozen RETFound ViT-L/16 MAE backbone once per process."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{BACKBONE_NAME}:{device}"
    if key not in _MODEL_CACHE:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(repo_id=RETFOUND_HF_REPO_ID, filename=RETFOUND_HF_FILENAME)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = _retfound_models.RETFound_mae(
            img_size=INPUT_SIZE, num_classes=5, drop_path_rate=0, global_pool=True
        )
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        # Expected mismatches, confirmed against this exact checkpoint:
        #  - missing "head.*": the throwaway num_classes=5 head is never used
        #    (only forward_features is called).
        #  - missing "fc_norm.*": global_pool=True adds a *new* LayerNorm over
        #    the mean-pooled patch tokens that this MAE-pretraining checkpoint
        #    never had; it keeps PyTorch's default LayerNorm init (weight=1,
        #    bias=0), which is exactly what RETFound's own latent_feature.ipynb
        #    silently relies on (same prepare_model() call, same strict=False).
        #  - unexpected "decoder_*", "mask_token", "decoder_pos_embed": the MAE
        #    decoder, discarded for linear probing.
        #  - unexpected "norm.*": global_pool=True deletes self.norm in favour
        #    of fc_norm (see models_vit.VisionTransformer.__init__), so the
        #    checkpoint's encoder-output norm is correctly unused here.
        expected_missing = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
        expected_unexpected_prefixes = ("decoder_", "mask_token", "norm.")
        unexpected_backbone_keys = [
            k for k in msg.unexpected_keys if not k.startswith(expected_unexpected_prefixes)
        ]
        missing_backbone_keys = [k for k in msg.missing_keys if k not in expected_missing]
        if unexpected_backbone_keys or missing_backbone_keys:
            raise RuntimeError(
                "RETFound checkpoint did not load cleanly into the backbone: "
                f"missing={missing_backbone_keys} unexpected={unexpected_backbone_keys}"
            )
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key], device


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    """Resize to 224x224 and standardize per image, per channel (RETFound's
    own ``latent_feature.ipynb`` preprocessing), not ImageNet mean/std."""
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    out = np.empty_like(arr)
    for c in range(3):
        channel = arr[..., c]
        mean = float(channel.mean())
        std = float(channel.std())
        out[..., c] = (channel - mean) / std if std > 1e-8 else channel - mean
    return np.transpose(out, (2, 0, 1))  # CHW


def _forward_pooled(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """``RETFound_mae(..., global_pool=True)``'s ``forward_features`` already
    returns the globally-pooled, LayerNorm'd ([fc_norm]) feature with shape
    [B, 1, D] (see models_vit.VisionTransformer.forward_features). Squeeze
    only that singleton middle dim -- unlike the notebook's bare
    ``torch.squeeze``, which silently also collapses the batch dimension
    when B == 1 and would corrupt single-image batches."""
    features = model.forward_features(batch)
    return features.squeeze(1)


def embed_rgb(rgb: np.ndarray, *, device: str | None = None) -> np.ndarray:
    """Embed a single in-memory RGB array."""
    return embed_rgb_batch([rgb], device=device)[0]


def embed_rgb_batch(images: list[np.ndarray], *, device: str | None = None) -> np.ndarray:
    """Embed a list of in-memory RGB arrays in one forward pass."""
    model, device = load_backbone(device)
    batch = torch.from_numpy(np.stack([_preprocess(rgb) for rgb in images])).to(device)
    with torch.no_grad():
        return _forward_pooled(model, batch.float()).cpu().numpy()


def _read_and_preprocess(path: str | Path) -> np.ndarray | None:
    try:
        return _preprocess(read_rgb(path))
    except Exception:  # noqa: BLE001 - unreadable file, caller marks ok=False
        return None


def embed_images(
    paths: list[str | Path],
    *,
    batch_size: int = 64,
    device: str | None = None,
    verbose: bool = False,
    io_workers: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, EMBED_DIM], ok [N] bool). Unreadable images get
    a zero row and ok=False, matching ``backbone.py``'s tolerant batch style.

    Default batch_size is smaller than DINOv2 ViT-B/14's (96 -> 64) because
    RETFound is ViT-Large (24 layers, 1024-dim) and the GPU is shared with
    other concurrent jobs on this machine; reduce further on CUDA OOM.

    Disk read + decode is overlapped with the GPU forward pass on a thread
    pool, exactly as in ``backbone.py``."""
    model, device = load_backbone(device)
    n = len(paths)
    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)

    batch_starts = list(range(0, n, batch_size))

    def submit_batch(pool: ThreadPoolExecutor, start: int):
        chunk = paths[start : start + batch_size]
        futures = [pool.submit(_read_and_preprocess, path) for path in chunk]
        return start, futures

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        pending = submit_batch(pool, batch_starts[0]) if batch_starts else None

        for i, start in enumerate(batch_starts):
            start, futures = pending
            results = [future.result() for future in futures]

            if i + 1 < len(batch_starts):
                pending = submit_batch(pool, batch_starts[i + 1])

            tensors, indices = [], []
            for offset, tensor in enumerate(results):
                if tensor is not None:
                    tensors.append(tensor)
                    indices.append(start + offset)

            if tensors:
                batch = torch.from_numpy(np.stack(tensors)).to(device, non_blocking=True)
                with torch.no_grad():
                    features = _forward_pooled(model, batch.float()).cpu().numpy()
                for row, index in zip(features, indices):
                    embeddings[index] = row
                    ok[index] = True

            if verbose:
                done = min(start + batch_size, n)
                print(f"  embedded {done}/{n}", flush=True)

    return embeddings, ok


if __name__ == "__main__":
    # Self-test: confirm the empirical embedding dimension rather than
    # assuming it, as called for in this module's docstring.
    model, device = load_backbone()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
        out = _forward_pooled(model, dummy)
    print(f"device={device} output shape={tuple(out.shape)} (expected [2, EMBED_DIM={EMBED_DIM}])")
    assert out.shape == (2, EMBED_DIM), f"EMBED_DIM constant ({EMBED_DIM}) does not match actual output dim {out.shape[-1]}"
    print("OK")
