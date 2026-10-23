"""Extract and cache frozen DINOv2 embeddings for the development set and
every validation cohort. Run once; downstream training/validation scripts
read the cached .npy files."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
from eyeqc_v2.backbone import embed_images

SOURCE_LABELS = REPO_ROOT / "data_final" / "source_labels"
CACHE = REPO_ROOT / "data_final" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "development": SOURCE_LABELS / "development_features.parquet",  # EyeQ train, used to fit the v3 head
    "EyeQ-test": SOURCE_LABELS / "image_level_EyeQ-test.parquet",
    "DRIMDB": SOURCE_LABELS / "image_level_DRIMDB.parquet",
    "FQS": SOURCE_LABELS / "image_level_FQS.parquet",
    "MSHF-CFP": SOURCE_LABELS / "image_level_MSHF-CFP.parquet",
    "DeepDRiD": SOURCE_LABELS / "image_level_DeepDRiD.parquet",
}


def main() -> None:
    for name, path in DATASETS.items():
        embed_path = CACHE / f"{name}_dinov2_embeddings.npy"
        ok_path = CACHE / f"{name}_dinov2_ok.npy"
        if embed_path.exists() and ok_path.exists():
            print(f"[{name}] cache already present, skipping")
            continue

        frame = pd.read_parquet(path)
        frame = frame[frame["image_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
        print(f"[{name}] embedding {len(frame)} images...")

        start = time.perf_counter()
        embeddings, ok = embed_images(frame["image_path"].tolist(), batch_size=128, verbose=True)
        elapsed = time.perf_counter() - start
        print(f"[{name}] done in {elapsed:.1f}s ({len(frame) / elapsed:.1f} img/s), {ok.sum()}/{len(frame)} readable")

        np.save(embed_path, embeddings)
        np.save(ok_path, ok)
        frame.to_parquet(CACHE / f"{name}_index.parquet", index=False)

    print("all embeddings cached in", CACHE)


if __name__ == "__main__":
    main()
