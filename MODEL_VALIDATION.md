# Model parity and safety gate

The available EyeQC research extractor calculates ten interpretable image-quality measurements and a six-by-six local-readability map from retinal photographs.

The frozen 778-feature classifier additionally requires the **exact** published 768-dimensional DINOv2 ViT-B/14 embedding. The supplied original research source references `eyeqc_v2.backbone` but does not contain that module. Guessing resize, crop, normalization, token aggregation, or checkpoint version could alter the calibrated probability and PASS/REVIEW/FAIL output.

**Consequently, the public Hugging Face tool computes the Python image measurements and real local maps but leaves the published calibrated EyeQC risk score disabled.** The repository contains an experimental candidate encoder behind an opt-in flag, disabled by default. Do not use it for clinical or manuscript-level results.

Before enabling published risk inference, recover the original preprocessing and weight checksum, regenerate embeddings for archived reference photographs, compare model probabilities to the original stored values, and confirm image-level and cohort-level parity within prespecified tolerances. Store this test and its expected reference vectors in the repository.

FLAIR and RRWNet provide independent outputs and are not trained or calibrated components of the published EyeQC classifier. Each requires its own model dependencies and weights.
