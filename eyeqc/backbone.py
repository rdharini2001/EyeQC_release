"""Frozen DINOv2 ViT-B/14 encoder used by EyeQC.

The photograph is resized to 224 x 224 pixels (area interpolation), normalised with the ImageNet mean and standard
deviation, and passed through the frozen encoder; the 768-dimensional class-token embedding is returned.  These steps
must not change: the stored classifier was fitted on embeddings produced exactly this way.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from .pipeline import read_rgb

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224
EMBED_DIM = 768
BACKBONE_NAME = "dinov2_vitb14"

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def load_backbone(device: str | None = None) -> tuple[torch.nn.Module, str]:
    """Load the frozen DINOv2 ViT-B/14 backbone once per process.

    ``device`` defaults to the environment variable ``EYEQC_DEVICE`` if set, otherwise to a GPU when one is available.
    """
    device = device or os.getenv("EYEQC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    key = f"{BACKBONE_NAME}:{device}"
    if key not in _MODEL_CACHE:
        model = torch.hub.load("facebookresearch/dinov2", BACKBONE_NAME, pretrained=True)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key], device


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normalized, (2, 0, 1))  # CHW


def embed_rgb(rgb: np.ndarray, *, device: str | None = None) -> np.ndarray:
    """Embed a single in-memory RGB array (for example an upload)."""
    return embed_rgb_batch([rgb], device=device)[0]


def embed_rgb_batch(images: list[np.ndarray], *, device: str | None = None) -> np.ndarray:
    """Embed a list of in-memory RGB arrays (e.g. synthetic degradation
    variants that never touch disk) in one forward pass."""
    model, device = load_backbone(device)
    batch = torch.from_numpy(np.stack([_preprocess(rgb) for rgb in images])).to(device)
    with torch.no_grad():
        return model(batch).cpu().numpy()


def _read_and_preprocess(path: str | Path) -> np.ndarray | None:
    try:
        return _preprocess(read_rgb(path))
    except Exception:  # noqa: BLE001 - unreadable file, caller marks ok=False
        return None


def embed_images(
    paths: list[str | Path],
    *,
    batch_size: int = 96,
    device: str | None = None,
    verbose: bool = False,
    io_workers: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [N, 768], ok [N] bool). Unreadable images get a zero
    row and ok=False rather than raising.

    Disk read + JPEG/PNG decode dominates wall time far more than the GPU
    forward pass at this batch size, so file decoding for the *next* batch is
    prefetched on a thread pool while the GPU works on the current one
    (cv2's decode/resize release the GIL, so this genuinely overlaps)."""
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

            # kick off decoding the next batch while this one runs on the GPU
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
                    features = model(batch).cpu().numpy()
                for row, index in zip(features, indices):
                    embeddings[index] = row
                    ok[index] = True

            if verbose:
                done = min(start + batch_size, n)
                print(f"  embedded {done}/{n}", flush=True)

    return embeddings, ok
