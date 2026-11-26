"""Embed a single image in a brand-new Python process and dump to .npy.
Used to test cross-process determinism (fresh CUDA context / cuDNN algo
selection each time), not just within-process repeat calls."""
import sys
from pathlib import Path
import os

REPOSITORY = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("EYEQC_DATASETS", REPOSITORY / "datasets"))
EXTERNAL = Path(os.environ.get("EYEQC_EXTERNAL", REPOSITORY / "external"))      # clones and weights of third-party methods
import numpy as np

CODE_ROOT = REPOSITORY
sys.path.insert(0, str(CODE_ROOT))
from eyeqc.backbone import embed_images  # noqa: E402

img_path = sys.argv[1]
out_path = sys.argv[2]
emb, ok = embed_images([img_path], batch_size=1, verbose=False)
assert ok[0]
np.save(out_path, emb[0])
print("done", emb[0][:5])
