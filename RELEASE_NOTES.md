# EyeQC research website release

A new GitHub Pages edition for transparent exploration of the supplied EyeQC results.

- Replaced the server-dependent first page with a responsive, static research website.
- Added authentic image-level local-readability maps calculated from the supplied photographs using the original Python extractor.
- Added nine simulated acquisition defects at five doses, each with recomputed interpretable measurements and downloadable dose tables.
- Added the five-cohort study overview and an eight-method AUROC/AUPRC/Brier comparison derived from the supplied result CSV files.
- Added offline photograph preview, explicitly not the validated EyeQC risk model.
- Added an optional endpoint connector for the Python ten-feature extractor, vessel module and FLAIR comparator.
- Added GitHub Pages deployment workflow and branch-based deployment instructions.

Remaining step before full model inference: recover and verify the original frozen DINOv2 preprocessing implementation against archived embeddings and scores.
