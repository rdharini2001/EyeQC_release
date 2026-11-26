---
title: EyeQC
emoji: 👁️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
python_version: 3.10.13
short_description: Interpretable fundus image quality assessment
---

# EyeQC

Calibrated PASS / REVIEW / FAIL triage, ten named quality measurements, their contributions, defect-specific maps and a
response to a quality question typed in your own words, for colour fundus photographs. The first tab runs the exact
model of the paper (`models/eyeqc_v3_model.json` with a frozen DINOv2 ViT-B/14 encoder).

Further tabs: comparison of two photographs, robustness and localisation, cohort analysis, harmonization, local
recalibration, reader agreement and the study results.

![EyeQC workflow](assets/overview.png)

![EyeQC interface and research applications](assets/interface_overview.png)

**Research use only.** Not a medical device. Do not upload identifiable patient data.

Code, tests and reproduction scripts: https://github.com/rdharini2001/EyeQC_release

Example photographs: DeepDRiD, CC BY-SA 4.0 (see `examples/ATTRIBUTION.md`). The "Vessels & FLAIR" tab is not part of
EyeQC or the paper: it uses RRWNet (code MIT; **weights CC BY-NC 4.0, non-commercial**) and FLAIR under their own terms.
