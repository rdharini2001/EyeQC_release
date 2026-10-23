# Deployment and release checks

The source app at `https://huggingface.co/spaces/Dharini24/EyeQC` is a Gradio Space (metadata checked October 2026). This release is a separate FastAPI/Docker site, so deploy it to a new Docker Space first.

1. Run `python -m pytest -q` and `uvicorn portal.app:app --port 8000`. Confirm sample uploads and actual tile maps work.
2. Build `docker build -t eyeqc-research .`; run locally and verify `/health`, `/api/check`, `/api/degradation`.
3. Create a new Hugging Face Docker Space (SDK: Docker), or deploy the Dockerfile to your host. Copy the repository contents, replacing the root `README.md` metadata header with the YAML in `hf_space/README.md` as needed; Hugging Face Docker Spaces use port 7860.
4. Keep `EYEQC_ENABLE_CANDIDATE_INFERENCE=0` until original `eyeqc_v2.backbone` or exact preprocessing is recovered and evaluated against reference embeddings on the same photographs.
5. Pin external FLAIR / vessel checkpoint revisions and package dependencies. Make optional modules return 503 when unavailable; they must never overwrite the paper's model output.
6. Configure maximum upload size and reverse proxy limits; decide whether PHI-containing images may be uploaded; set appropriate log retention, access controls and consent procedures.
7. Verify CPU memory footprint, DINO model download and startup latency on the actual target host. The 208-ms historical benchmark must not be described as current site latency without a new benchmark.
8. When publishing to GitHub, replace placeholder manuscript citation, add an accessible model-checksum manifest and ensure sample image redistribution is permitted.

**Safety note:** A public research website is not a patient-care device. It must not automatically accept, reject, or diagnose clinical images. No prospective reduction in reader variation, unnecessary reacquisitions, or patient-level benefit has been established.
