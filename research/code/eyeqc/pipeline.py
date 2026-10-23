"""Image-manifest extraction, scoring, and export."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .core import FEATURES, EyeQCModel, extract_features, scope_check


def read_rgb(path: str | Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _extract_one(path: str) -> dict:
    """Extract the ten hand-crafted features for one image.

    Catches any per-image failure (corrupted file, truncated JPEG,
    undecodable format) and returns an all-NaN row with ``extract_ok=False``
    instead of raising, matching eyeqc_v2.backbone.embed_images' ok-mask
    pattern. Without this, one bad file anywhere in a manifest kills the
    entire joblib batch rather than just that image -- a real deployment
    risk for a quality-triage tool meant to run unattended on an imaging
    queue.
    """
    try:
        features = extract_features(read_rgb(path))
        scope = scope_check(features)
        return {name: features[name] for name in FEATURES} | {
            "runtime_feature_sec": features["runtime_feature_sec"],
            "retina_fraction": features["retina_fraction"],
            "readable_area_fraction": features["readable_area_fraction"],
            "circularity_audit": features["circularity_audit"],
            "borders_touched": features["borders_touched"],
            "cropped_or_zoomed": features["cropped_or_zoomed"],
            "in_scope": scope["in_scope"],
            "scope_reason": scope["reason"],
            "extract_ok": True,
            "extract_error": None,
        }
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure mode for one image must not sink the batch
        return {name: float("nan") for name in FEATURES} | {
            "runtime_feature_sec": float("nan"),
            "retina_fraction": float("nan"),
            "readable_area_fraction": float("nan"),
            "circularity_audit": float("nan"),
            "borders_touched": float("nan"),
            "cropped_or_zoomed": None,
            "in_scope": False,
            "scope_reason": f"feature extraction failed: {type(exc).__name__}: {exc}",
            "extract_ok": False,
            "extract_error": f"{type(exc).__name__}: {exc}",
        }


def extract_manifest(
    manifest: pd.DataFrame,
    cache_path: str | Path,
    *,
    workers: int = 4,
    force: bool = False,
) -> pd.DataFrame:
    """Extract the ten features in manifest order and cache them as Parquet."""

    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    data = manifest.copy().reset_index(drop=True)
    missing = ~data["image_path"].map(lambda value: Path(value).exists())
    if missing.any():
        names = data.loc[missing, "image_path"].head(5).tolist()
        raise FileNotFoundError(f"{missing.sum()} images are missing; first paths: {names}")

    features = Parallel(n_jobs=workers, prefer="processes", batch_size=16)(
        delayed(_extract_one)(path) for path in data["image_path"]
    )
    output = pd.concat([data, pd.DataFrame(features)], axis=1)
    n_failed = int((~output["extract_ok"]).sum())
    if n_failed:
        failed_paths = output.loc[~output["extract_ok"], "image_path"].head(5).tolist()
        print(
            f"extract_manifest: {n_failed}/{len(output)} images failed feature extraction "
            f"(kept as NaN rows, excluded downstream by the existing notna() ok-mask convention); "
            f"first failing paths: {failed_paths}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(cache_path, index=False)
    return output


def score_frame(frame: pd.DataFrame, model: EyeQCModel) -> pd.DataFrame:
    rows = [model.score_features(row) for row in frame[list(FEATURES)].to_dict("records")]
    scores = pd.DataFrame(rows)
    output = pd.concat(
        [
            frame.reset_index(drop=True),
            scores.drop(columns=["axis_contributions"]).reset_index(drop=True),
        ],
        axis=1,
    )
    for axis in FEATURES:
        output[f"contribution_{axis}"] = [row[axis] for row in scores["axis_contributions"]]
    return output


def benchmark_throughput(
    image_paths: list[str] | list[Path],
    *,
    worker_grid: tuple[int, ...] = (1, 2, 4),
    repeats: int = 1,
    prefer: str = "processes",
) -> pd.DataFrame:
    """Measure feature-extraction throughput at increasing worker counts.

    This mirrors the exact call the GUI batch tab and ``extract_manifest`` make
    (``read_rgb`` then ``extract_features``) so the reported images-per-second
    figure is a deployment-realistic number rather than a microbenchmark.
    """

    paths = [str(path) for path in image_paths]
    if not paths:
        raise ValueError("benchmark_throughput requires at least one image path")

    rows = []
    for workers in worker_grid:
        for repeat in range(repeats):
            start = time.perf_counter()
            Parallel(n_jobs=workers, prefer=prefer, batch_size=8)(
                delayed(extract_features)(read_rgb(path)) for path in paths
            )
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "workers": workers,
                    "repeat": repeat,
                    "images": len(paths),
                    "wall_time_sec": elapsed,
                    "images_per_sec": len(paths) / elapsed if elapsed > 0 else float("nan"),
                    "mean_latency_ms": elapsed / len(paths) * 1000 if paths else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def latency_percentiles(frame: pd.DataFrame, column: str = "runtime_feature_sec") -> pd.Series:
    """Summarize per-image processing latency as clinic-relevant percentiles."""

    values = frame[column].dropna().to_numpy(float) * 1000
    if values.size == 0:
        raise ValueError(f"No finite values in column {column!r}")
    percentiles = {f"p{q}_ms": float(np.percentile(values, q)) for q in (50, 90, 95, 99)}
    return pd.Series({"n": values.size, "mean_ms": float(values.mean()), **percentiles, "max_ms": float(values.max())})


def contract_sha256(model_path: str | Path) -> str:
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()


def save_provenance(model_path: str | Path, output_path: str | Path) -> None:
    contract = json.loads(Path(model_path).read_text(encoding="utf-8"))
    record = {
        "model_version": contract["version"],
        "feature_hash": contract["feature_hash"],
        "model_sha256": contract_sha256(model_path),
        "features": contract["features"],
        "pass_risk_threshold": contract["pass_risk_threshold"],
        "fail_risk_threshold": contract["fail_risk_threshold"],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

