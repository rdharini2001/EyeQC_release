<div align="center">

# EyeQC: A research framework for retinal photograph quality assessment, spatial measurements and controlled acquisition experiments.**

[Explore the research site](https://rdharini2001.github.io/EyeQC_release/) · [Read the Methods](research/MANUSCRIPT_ADDITIONS.md) · [Deployment guide](docs/DEPLOY.md) · [Model validation](MODEL_VALIDATION.md)

![Actual retinal photographs, six-by-six tile measurements and measured spatial maps](docs/assets/figures/eyeqc_spatial_atlas.webp)

</div>

---

## What EyeQC measures

EyeQC was developed to make the technical quality of color fundus photographs easier to inspect and compare. The study combines **ten interpretable image measurements** with a frozen **768-dimensional DINOv2 ViT-B/14 embedding** in a calibrated logistic regression classifier. It also records local readability measurements that can be examined as tile grids or interpolated maps.

The public research site shows the measurements alongside the photographs that produced them. The code includes the original extraction and degradation routines, the frozen classifier contract, analysis scripts, and two **separate optional modules** for retinal vessel structure (RRWNet) and FLAIR image-description comparisons.

### Results reported in the manuscript

| External or held-out cohort | Photographs | EyeQC AUROC |
| --- | ---: | ---: |
| EyeQ-test · held-out same-source | 16,249 | 0.984 |
| DRIMDB | 194 | 1.000 |
| FQS | 2,246 | 0.897 |
| MSHF-CFP | 802 | 0.939 |
| DeepDRiD | 1,600 | 0.780 |

**21,091** evaluation photographs across five cohorts. Reader agreement ranged from weighted κ **0.702–0.777** in a separate 80-image study. Median CPU processing time in the original benchmark was **208 ms**. These figures are archived research findings, **not** performance measured by the website on a visitor's uploaded photograph.

## Explore the website

| Workspace | What happens | Where the output comes from |
| --- | --- | --- |
| **Quality workbench** | Choose a photograph; inspect its ten measurements, six-by-six tiles, map, overlay and downloadable record | Python EyeQC feature extractor applied to the included photographs |
| **Vessel view** | Explore green-channel vessel contrast; optionally run RRWNet to see separate artery, vein and vessel probability maps | Contrast: OpenCV on the actual photograph. RRWNet: pretrained model via Python server |
| **FLAIR comparison** | Run quality-description prompts and inspect per-prompt similarities | Separate FLAIR model via Python server; *not* calibrated EyeQC risk |
| **Degradation laboratory** | Change defect type, severity or photograph; inspect the altered photograph, local measurements and dose-response curve | Original degradation and feature extraction functions, run on the supplied photographs |
| **Spatial atlas** | Inspect four-panel retinal quality figures and controlled-defect examples | Rebuildable figures from `scripts/build_visual_atlas.py` |
| **Evidence explorer** | Compare discrimination, calibration, readers and other cohorts | Aggregated values from the research analysis archive |

![Controlled shadow, glare and local blur applied to retinal photographs and measured again](docs/assets/figures/eyeqc_degradation_atlas.webp)

The atlas uses a fixed 0–1 display scale. The spatial measurements are **interpolated local-readable tiles**, not pixel-level expert annotations or attribution maps for the frozen DINOv2 encoder.

## One interface, two deployment choices

**GitHub Pages:** publish [`docs/`](docs/) directly. The gallery, measurable tile maps, nine-defect laboratory, charts, figures, downloads and browser-only photographic preview work without a server. Full Python computation for arbitrary uploaded photographs and pretrained FLAIR/RRWNet inference require a running API service.

**Python-backed website:** deploy the included FastAPI server. It serves the *same website* at `/`, with `POST /api/check`, `/api/degradation`, `/api/dose-curve`, `/api/batch`, `/api/vessel`, and `/api/flair`. There is no separate visual interface to visit. The live model features run inside the existing website after it automatically connects to its own server.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn portal.app:app --host 127.0.0.1 --port 7860
# Open http://127.0.0.1:7860
```

To enable RRWNet and FLAIR, install optional dependencies and make their published weights accessible to the service. FLAIR's upstream package is installed separately; see [the module guidance](docs/DEPLOY.md). For a full dependency image, see `Dockerfile.full`. The server reports missing weights/dependencies instead of silently replacing the model with an invented result.

```bash
pip install -r requirements-optional.txt
# Then install the retinal FLAIR package: `pip install git+https://github.com/jusiro/FLAIR.git`.
```

For Docker deployment, `docker build -t eyeqc .` and `docker run -p 7860:7860 eyeqc`. Optional large model weights must be made available at runtime and may require a machine with sufficient memory.

## Reproduce the figures

Only use photographs you have permission to redistribute. The figure generator uses the repository's measured gallery tile arrays, recorded code-generated dose examples and supplied retinal photographs:

```bash
python scripts/check_pages.py
python scripts/build_visual_atlas.py
```

The generated figures are saved to `docs/assets/figures/`, with the auxiliary contrast overlays in `docs/assets/vessels/`. Rebuilding the main research gallery from source photographs uses `scripts/build_assets.py` and `eyeqc.core.extract_features`.

## Code map

```text
EyeQC_release/
├── docs/                 Research website; images, measured maps, dose frames and figures
├── eyeqc/                Core extraction and controlled image degradation
├── eyeqc_runtime/        Frozen trained-head contract; encoder parity gate
├── eyeqc_vessel/         Optional RRWNet vessel inference
├── eyeqc_flair/          Optional FLAIR image–text analysis
├── portal/               Same-site FastAPI image and model endpoints
├── models/               Supplied classifier contracts and checksums
├── research/             Study analyses and aggregate data (no individual patient records)
├── scripts/              Rebuild figures, prepare assets, check website
├── tests/                Model-contract, image, server and release tests
└── .github/workflows/    GitHub Pages publication and CI
```

## Model verification and intended use

The provided source code does **not** include the original exact `eyeqc_v2.backbone` preprocessing module needed to reproduce the published DINOv2 embeddings. Until that module and its weight revision are recovered and tested against archived embeddings, the online website **must not display invented calibrated EyeQC risk probabilities or clinical PASS/REVIEW/FAIL recommendations**. The frozen logistic model contract is retained for parity testing; the candidate encoder is disabled by default.

The included vessel and FLAIR tools are **not part of the trained 778-feature EyeQC model** and have not been validated in the EyeQC manuscript. The project is intended for research demonstration and method development, not autonomous image acceptance or patient-care decisions.

Before publishing this repository, verify rights to share the supplied photographs, original dataset restrictions, third-party model licenses and the study's ethics/data availability statements. No patient images or uploaded file contents are saved by the API.

## Citation

See [`CITATION.cff`](CITATION.cff). The manuscript is in preparation; add the final publication DOI, repository release tag and approved data citation after acceptance.
