"""Extract and cache frozen RETFound (ViT-L/16 MAE, retina-specific) embeddings
for the development set and every external validation cohort, mirroring
``extract_embeddings.py`` (the DINOv2 equivalent) but reading image paths
straight from the already-built ``{name}_index.parquet`` caches rather than
re-deriving them from source labels, since those index files -- and their
DINOv2 embeddings/ok masks -- already exist and must not be touched.

The GPU is shared with two other concurrent jobs on this machine, so this
script processes each cohort in resumable chunks and checkpoints partial
progress to disk (``{name}_retfound_embeddings.partial.npy`` /
``.ok.partial.npy`` / ``.progress.json`` in the cache dir) rather than
holding everything only in memory for the whole cohort. If the process is
killed (CUDA OOM, or a hard crash from GPU contention) partway through a
cohort, re-running this script resumes from the last completed chunk instead
of starting that cohort over. On an OOM inside a chunk, the chunk's batch
size is halved and retried before giving up and marking those images
ok=False. Partial/checkpoint files are removed once a cohort's real output
files (``{name}_retfound_embeddings.npy`` / ``{name}_retfound_ok.npy``) are
written.

Run once (or resume as many times as needed); ``train_v3_retfound.py`` /
``run_validation_v3_retfound.py`` read the final cached .npy files this
script writes."""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone_retfound import EMBED_DIM, embed_images  # noqa: E402

CACHE = REPO_ROOT / "data_final" / "cache"

COHORTS = ["development", "EyeQ-test", "DRIMDB", "FQS", "MSHF-CFP", "DeepDRiD"]

CHUNK_SIZE = 200  # checkpoint cadence: images per chunk
INITIAL_BATCH_SIZE = 16  # conservative given GPU is shared with 2 other concurrent jobs
MIN_BATCH_SIZE = 1
MAX_WAIT_ROUNDS = 20  # each round: shrink to MIN_BATCH_SIZE, then back off and retry from INITIAL_BATCH_SIZE
BACKOFF_SECONDS = 30.0
IO_WORKERS = 8


MIN_FREE_VRAM_BYTES = 2 * 1024 ** 3  # wait for >=2GB free before even attempting a chunk


def _wait_for_free_vram() -> None:
    if not torch.cuda.is_available():
        return
    for wait_round in range(MAX_WAIT_ROUNDS):
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if free_bytes >= MIN_FREE_VRAM_BYTES:
            return
        print(
            f"    only {free_bytes / 1024**3:.2f}GB free on the shared GPU (other jobs active); "
            f"waiting {BACKOFF_SECONDS:.0f}s before attempting the next chunk",
            flush=True,
        )
        time.sleep(BACKOFF_SECONDS)


def _embed_chunk_with_oom_backoff(paths: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Embed one chunk, halving batch_size on CUDA OOM until it fits. If even
    MIN_BATCH_SIZE OOMs (the other two concurrent jobs on this shared GPU are
    at a usage peak), back off for BACKOFF_SECONDS and retry the whole chunk
    from INITIAL_BATCH_SIZE again, up to MAX_WAIT_ROUNDS times, before giving
    up and marking those images ok=False (matching the tolerant-batch
    convention used throughout this repo)."""
    for wait_round in range(MAX_WAIT_ROUNDS):
        _wait_for_free_vram()
        size = batch_size
        while True:
            try:
                return embed_images(paths, batch_size=size, device=None, verbose=False, io_workers=IO_WORKERS)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if size <= MIN_BATCH_SIZE:
                    break
                size = max(MIN_BATCH_SIZE, size // 2)
                print(f"    CUDA OOM, retrying chunk with batch_size={size}", flush=True)
        print(
            f"    CUDA OOM even at batch_size={MIN_BATCH_SIZE} (other GPU jobs at a usage peak); "
            f"backing off {BACKOFF_SECONDS:.0f}s (wait round {wait_round + 1}/{MAX_WAIT_ROUNDS})",
            flush=True,
        )
        time.sleep(BACKOFF_SECONDS)

    print(f"    gave up after {MAX_WAIT_ROUNDS} backoff rounds; marking this chunk's {len(paths)} images as failed")
    return np.zeros((len(paths), EMBED_DIM), dtype=np.float32), np.zeros(len(paths), dtype=bool)


def _progress_path(name: str) -> Path:
    return CACHE / f"{name}_retfound_embeddings.progress.json"


def _partial_embed_path(name: str) -> Path:
    return CACHE / f"{name}_retfound_embeddings.partial.npy"


def _partial_ok_path(name: str) -> Path:
    return CACHE / f"{name}_retfound_ok.partial.npy"


def process_cohort(name: str) -> None:
    final_embed_path = CACHE / f"{name}_retfound_embeddings.npy"
    final_ok_path = CACHE / f"{name}_retfound_ok.npy"
    if final_embed_path.exists() and final_ok_path.exists():
        print(f"[{name}] RETFound cache already present, skipping")
        return

    index_path = CACHE / f"{name}_index.parquet"
    frame = pd.read_parquet(index_path)
    paths = frame["image_path"].tolist()
    n = len(paths)

    progress_path = _progress_path(name)
    partial_embed_path = _partial_embed_path(name)
    partial_ok_path = _partial_ok_path(name)

    if progress_path.exists() and partial_embed_path.exists() and partial_ok_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_start = int(progress["next_start"])
        embeddings = np.load(partial_embed_path)
        ok = np.load(partial_ok_path)
        if embeddings.shape != (n, EMBED_DIM) or ok.shape != (n,):
            print(f"[{name}] partial cache shape mismatch, restarting cohort from scratch")
            next_start = 0
            embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
            ok = np.zeros(n, dtype=bool)
        else:
            print(f"[{name}] resuming from image {next_start}/{n}")
    else:
        next_start = 0
        embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)
        ok = np.zeros(n, dtype=bool)

    print(f"[{name}] embedding {n} images with RETFound (dim={EMBED_DIM}), starting at {next_start}...")
    start_time = time.perf_counter()

    for chunk_start in range(next_start, n, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n)
        chunk_paths = paths[chunk_start:chunk_end]
        chunk_embeddings, chunk_ok = _embed_chunk_with_oom_backoff(chunk_paths, INITIAL_BATCH_SIZE)
        embeddings[chunk_start:chunk_end] = chunk_embeddings
        ok[chunk_start:chunk_end] = chunk_ok

        np.save(partial_embed_path, embeddings)
        np.save(partial_ok_path, ok)
        progress_path.write_text(json.dumps({"next_start": chunk_end, "n": n}), encoding="utf-8")

        elapsed = time.perf_counter() - start_time
        rate = (chunk_end - next_start) / elapsed if elapsed > 0 else float("nan")
        print(f"[{name}] checkpointed {chunk_end}/{n} ({rate:.1f} img/s since resume)", flush=True)

    elapsed = time.perf_counter() - start_time
    print(f"[{name}] done in {elapsed:.1f}s, {int(ok.sum())}/{n} embedded ok, {int((~ok).sum())} failed")
    if (~ok).any():
        failed_paths = frame.loc[~ok, "image_path"].head(10).tolist()
        print(f"[{name}] sample failed paths: {failed_paths}")

    np.save(final_embed_path, embeddings)
    np.save(final_ok_path, ok)
    print(f"[{name}] wrote {final_embed_path.name} and {final_ok_path.name}")

    for p in (progress_path, partial_embed_path, partial_ok_path):
        p.unlink(missing_ok=True)


def main() -> None:
    for name in COHORTS:
        process_cohort(name)
    print("all RETFound embeddings cached in", CACHE)


if __name__ == "__main__":
    main()
