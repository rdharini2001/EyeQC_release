"""Realistic end-to-end latency/throughput for the v3 pipeline: read image,
compute the 10 hand features, embed with DINOv2, score — the exact sequence
a live upload goes through, measured both as single-image latency (what a
clinician sees for one photograph) and as GPU-batched throughput (what a
batch upload sees)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_rgb, embed_rgb_batch  # noqa: E402
from eyeqc.core import extract_features  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

TABLES = REPO_ROOT / "data_final" / "tables"
N_LATENCY = 60
N_BATCH = 256


def main() -> None:
    frame = pd.read_parquet(TABLES / "image_level_v3_EyeQ-test.parquet").sample(n=N_LATENCY + N_BATCH, random_state=20260822)
    latency_paths = frame["image_path"].iloc[:N_LATENCY].tolist()
    batch_paths = frame["image_path"].iloc[N_LATENCY:].tolist()

    # warm up (first call pays CUDA/model init cost, exclude from measurement)
    warm_rgb = read_rgb(latency_paths[0])
    extract_features(warm_rgb)
    embed_rgb(warm_rgb)

    latencies_ms = []
    for path in latency_paths:
        t0 = time.perf_counter()
        rgb = read_rgb(path)
        extract_features(rgb)
        embed_rgb(rgb)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    latencies_ms = np.array(latencies_ms)

    print(f"single-photograph end-to-end latency over {N_LATENCY} real images:")
    print(f"  mean={latencies_ms.mean():.1f}ms  p50={np.percentile(latencies_ms,50):.1f}ms  "
          f"p90={np.percentile(latencies_ms,90):.1f}ms  p99={np.percentile(latencies_ms,99):.1f}ms")

    t0 = time.perf_counter()
    rgb_images = [read_rgb(p) for p in batch_paths]
    read_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for rgb in rgb_images:
        extract_features(rgb)
    features_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for start in range(0, len(rgb_images), 64):
        embed_rgb_batch(rgb_images[start:start + 64])
    embed_elapsed = time.perf_counter() - t0

    total_elapsed = read_elapsed + features_elapsed + embed_elapsed
    print(f"\nbatch throughput over {N_BATCH} real images (GPU-batched embedding, batch=64):")
    print(f"  read: {read_elapsed:.2f}s  features: {features_elapsed:.2f}s  embed: {embed_elapsed:.2f}s")
    print(f"  total: {total_elapsed:.2f}s = {N_BATCH/total_elapsed:.1f} images/s")

    summary = pd.DataFrame(
        [
            {"metric": "single_latency_mean_ms", "value": float(latencies_ms.mean())},
            {"metric": "single_latency_p50_ms", "value": float(np.percentile(latencies_ms, 50))},
            {"metric": "single_latency_p90_ms", "value": float(np.percentile(latencies_ms, 90))},
            {"metric": "single_latency_p99_ms", "value": float(np.percentile(latencies_ms, 99))},
            {"metric": "batch_images_per_sec", "value": N_BATCH / total_elapsed},
        ]
    )
    summary.to_csv(TABLES / "throughput_benchmark_v3.csv", index=False)
    print(f"\nwrote {TABLES / 'throughput_benchmark_v3.csv'}")


if __name__ == "__main__":
    main()
