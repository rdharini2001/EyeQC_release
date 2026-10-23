# Model parity and safety gate

The released Python feature extractor computes ten documented image-quality measurements and local six-by-six tile values.

The frozen hybrid classifier contract also needs the published frozen DINOv2 ViT-B/14 embedding. The uploaded source references `eyeqc_v2.backbone` but did not contain that exact preprocessing module; a guessed resize, crop, normalization, or token aggregation could change scores. Therefore the static Pages site does **not** calculate calibrated hybrid scores or PASS/REVIEW/FAIL categories from a newly uploaded image.

Before enabling full inference, recover the exact original encoder, normalization/transforms, model weight revision/checksum and embedding extraction method. Compare held-out archived embeddings and predictions against newly computed values using fixed tolerances; confirm cohort-level and image-level parity. The optional server contains an explicitly experimental candidate encoder behind an opt-in flag, disabled by default. Do not call its outputs the validated model until parity has passed.

FLAIR and vessel modules are optional additional analyses, not part of the manuscript's original validated classifier and not clinical recommendations.
