---
title: EyeQC — Retinal Image Quality Lab
emoji: 👁️
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 7860
fullWidth: true
header: mini
short_description: Explore interpretable fundus image quality, controlled degradations, vessel structure and FLAIR comparisons.
models:
- jusiro2/FLAIR
- j-morano/rrwnet-rite
---

# EyeQC — Retinal Image Quality Lab

An interactive research tool for retinal fundus photography. The site shows interpretable measurements, spatial quality maps, controlled acquisition defects, vessel analysis and a separate FLAIR image–text comparison, all in one place.

This app runs as a Docker Space with a Python backend. A photograph is sent to the Space only when a visitor explicitly runs an analysis. Uploaded images are processed in memory and are not stored by the app. Hugging Face platform handling and logs are governed by its own service policies.

## What each result means

- **EyeQC measurements:** Ten quality measurements computed from the photograph by the research extractor. The colored local-readability maps interpolate six-by-six measured tiles and are *not* pixel-level expert segmentation masks.
- **Controlled degradations:** Reproducible changes to the image using the research transformation code, with recalculated measurements.
- **Vessel structure:** Optional RRWNet artery, vein, and vessel-probability views. The checkpoint is downloaded from `j-morano/rrwnet-rite` on first use.
- **FLAIR:** Optional prompt-based comparison from the public FLAIR retinal vision–language model, `jusiro2/FLAIR`. The prompt scores are not calibrated quality probabilities.
- **Published EyeQC calibrated risk:** Disabled until the exact DINOv2 preprocessing used in the paper has been recovered and shown to reproduce the archived embeddings. We do not substitute a different encoder.

The website includes historical validation results from five cohorts, with an explanation of the limits of using operating thresholds at new sites.

**Research use only.** This app is not approved to decide whether a patient's retinal photograph should be accepted, rejected, or repeated.

Source code and reproducible analyses: [rdharini2001/EyeQC_release](https://github.com/rdharini2001/EyeQC_release).
