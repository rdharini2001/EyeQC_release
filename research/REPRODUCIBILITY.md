# Research reproducibility status

`code_final/` preserves the input scripts in their original relative layout, with `eyeqc/` copied under it as originally supplied. `data_final/` is not included in the public package: the user's source archive contained large participant/image-level tables and embedding caches. To rerun archived analyses, expand the authorized private `data_final.zip` into `data_final/` next to `code_final/`, confirm image paths resolve, install Parquet support, and recover the original `eyeqc_v2` code.

The 778-feature head contract is included and can be scored exactly from a ten-measure vector and a 768-feature cached embedding. However, the original `eyeqc_v2.backbone` implementation was not in the supplied code, so end-to-end image-to-score numerical parity **has not been established**. The included candidate encoder in `eyeqc_runtime/model.py` is clearly isolated and disabled by default. The original model's calibration parameters and thresholds are preserved without refitting.

Scientific checks to complete: compare frozen DINOv2 embeddings on identical images, verify mean absolute feature differences, verify predicted probabilities against archived per-image `risk_bad`, check threshold boundary behavior, and commit all model hashes and the definitive environment.
