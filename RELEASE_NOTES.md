# EyeQC release — Hugging Face workspace

This edition keeps the research repository and the live analysis workspace separate.

- The existing Python research extractor, studies, trained model contracts, selected results and paper-oriented figures are retained.
- The unified web interface is hosted through a **Hugging Face Docker Space** rather than GitHub Pages. Both the website and Python endpoints run in the same Space.
- Visitors can inspect supplied examples, compute ten measurements on uploaded images, view real six-by-six local-readability maps, apply nine controlled degradations, inspect five-dose curves and export batch measurements.
- The optional FLAIR image–text comparison and RRWNet artery/vein model are available as on-page tools when the respective upstream dependencies and weights load.
- The GitHub Pages deployment workflow is removed. The remaining GitHub Actions workflow tests the research code rather than publishing a website.
- The original DINOv2 end-to-end scoring pipeline is not claimed as reproduced; calibrated risk outputs remain disabled until encoder parity is checked.

Run `python scripts/package_space.py --output ../EyeQC_hf_space` to create the standalone Space release, and read `HUGGINGFACE_DEPLOY.md` before publishing.
