"""CPU throughput and per-image latency for the full v3 pipeline (read ->
ten axes -> DINOv2 embedding -> score), in the same 1/2/8-worker format as
the manuscript's Supp Table 8B, plus feature+embedding latency percentiles
across a large sample (paper's Panel D runtime numbers).

Forces CPU-only: the manuscript explicitly evaluates a "CPU-based
implementation" for real-time-at-acquisition feasibility on ordinary clinic
hardware, and this machine has a GPU that the backbone would otherwise pick
up automatically, giving a number that isn't the one the paper is making a
claim about (and isn't reproducible on a GPU-less clinic workstation)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must precede the first `import torch` anywhere in the process

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.hybrid_model import EyeQCModelV3  # noqa: E402
from eyeqc.pipeline import read_rgb  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(1)  # avoid PyTorch's own intra-op threads oversubscribing against our worker-level threading

TABLES = REPO_ROOT / "data_final" / "tables"
MODELS = REPO_ROOT / "data_final" / "models"
SOURCE_LABELS = REPO_ROOT / "data_final" / "source_labels"

WORKER_GRID = (1, 2, 8)
REPEATS = 1
SAMPLE_N = 200
LATENCY_N = 300
SEED = 20260822


def main() -> None:
    model = EyeQCModelV3(MODELS / "eyeqc_v3_model.json")

    scored = pd.read_parquet(TABLES / "image_level_v3_EyeQ-test.parquet")
    sample_paths = scored.sample(min(SAMPLE_N, len(scored)), random_state=SEED)["image_path"].tolist()

    # warm up (first call pays lazy CUDA/model init, excluded from measurement)
    warm = read_rgb(sample_paths[0])
    model.analyze(warm)

    rows = []
    for workers in WORKER_GRID:
        for repeat in range(REPEATS):
            start = time.perf_counter()
            # v3's per-image call already batches its own GPU op; parallelizing across
            # threads overlaps the CPU-bound extract_features + disk read portions,
            # the same axis v1's process-based benchmark parallelized.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda p: model.analyze(read_rgb(p)), sample_paths))
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "workers": workers, "repeat": repeat, "images": len(sample_paths),
                    "wall_time_sec": elapsed, "images_per_sec": len(sample_paths) / elapsed,
                    "mean_latency_ms": elapsed / len(sample_paths) * 1000,
                }
            )
            print(f"workers={workers} repeat={repeat}: {len(sample_paths)/elapsed:.2f} img/s", flush=True)

    throughput = pd.DataFrame(rows)
    throughput.to_csv(TABLES / "throughput_benchmark_v3_gridformat.csv", index=False)
    print("\n", throughput.round(3).to_string(index=False))

    # single-threaded per-image latency percentiles (matches the manuscript's
    # "median feature-extraction time" framing, extended to the full v3 pipeline)
    latency_paths = scored.sample(min(LATENCY_N, len(scored)), random_state=SEED + 1)["image_path"].tolist()
    latencies_ms = []
    for path in latency_paths:
        t0 = time.perf_counter()
        model.analyze(read_rgb(path))
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    latencies_ms = np.array(latencies_ms)
    percentiles = {
        "n": len(latencies_ms), "mean_ms": latencies_ms.mean(),
        "p50_ms": np.percentile(latencies_ms, 50), "p90_ms": np.percentile(latencies_ms, 90),
        "p95_ms": np.percentile(latencies_ms, 95), "p99_ms": np.percentile(latencies_ms, 99),
        "max_ms": latencies_ms.max(),
    }
    pd.Series(percentiles).to_frame("value").to_csv(TABLES / "feature_latency_percentiles_v3.csv")
    print("\nv3 end-to-end latency percentiles:")
    print(pd.Series(percentiles).round(1))


if __name__ == "__main__":
    main()
