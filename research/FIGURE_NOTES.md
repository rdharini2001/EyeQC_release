# Figure/data provenance

- Real patient retinal photos supplied for development of figure examples; public redistribution must be authorized.
- Six-by-six map: `eyeqc.core.extract_features` local-tile `quality` values, interpolated by `portal.spatial.make_layers`. Map RGB reflects the complement of readability; red means less locally readable.
- Dose lab: `eyeqc.experiments.apply_degradation` at 0, 0.25, 0.50, 0.75, 1.00 with `np.random.default_rng(20260822 + dose_index)`, recalculated using the same extractor.
- Predicted classifier probability and readers' labels are **not** being shown on the photo maps. The paper's published numbers are in `docs/assets/data/evidence.json` and are distinctly labelled as study summaries.
- The illustrative browser upload preview uses approximate Canvas luminance and local contrast/detail, separate from the Python feature extractor; it does not use or claim the validated EyeQC calibrated risk model.
