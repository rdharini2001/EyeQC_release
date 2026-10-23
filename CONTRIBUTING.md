# Contributing

Thank you for helping improve EyeQC. Please keep changes to the evaluated model separate from experimental additions. When changing the feature extractor, normalization or scoring head, document the model version and rerun the held-out parity checks. When adding figures, record the image source, code revision, seed, and exact plotted values. Do not publish photographs without the required permissions.

The public Pages site lives in `docs/`; Python research code lives in `eyeqc/` and `research/`; optional FLAIR/vessel functionality stays separate.

Before opening a pull request, run `python scripts/check_pages.py` and `node --check docs/assets/site.js`. If modifying the Python API, run the tests under `tests/` and verify an actual local fundus file returns ten measurements.
