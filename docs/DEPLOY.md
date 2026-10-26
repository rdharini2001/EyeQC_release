# Publish the EyeQC website

**Repository:** `rdharini2001/EyeQC_release`  
**Expected GitHub Pages site:** https://rdharini2001.github.io/EyeQC_release/

The site lives entirely in `/docs`. It contains actual retinal photographs, measured local quality maps, code-generated degradation frames and visualizations of archived results.

## Publish with GitHub Pages

1. Commit the contents of this release to the `main` branch of `rdharini2001/EyeQC_release`, keeping `/docs` at repository root.
2. In **Settings → Pages**, select **Deploy from a branch → `main` → `/docs` → Save**.
3. Alternatively, select **GitHub Actions** and let `.github/workflows/pages.yml` deploy `docs/` after every change. Choose one method, not both.
4. Visit `https://rdharini2001.github.io/EyeQC_release/` once the deployment completes. Hard-refresh if the earlier site is cached.

The current GitHub repository is a distinct project from the older `EyeQC` repository and the Hugging Face Space. GitHub Pages does not update those older URLs by itself. Internal website links point to `EyeQC_release`.

## Running the complete interface with Python models

GitHub Pages serves only HTML, JavaScript, and static files. It cannot execute Python or load Torch models. The **same website** is also served at `/` by the repository's FastAPI application:

```bash
pip install -r requirements.txt
uvicorn portal.app:app --host 0.0.0.0 --port 7860
```

The site detects a same-origin Python server automatically. A Pages deployment can talk to a separate API host without opening a different website: set `apiBase` in `docs/assets/config.json` to your HTTPS backend URL, for example `https://your-backend.example`. The image stays in the browser until the user explicitly runs a model. Add the GitHub Pages origin to `EYEQC_CORS_ORIGINS` on the API host if changing the default domain.

For a Docker-capable host or Hugging Face Docker Space using the core tools:

```bash
docker build -t eyeqc .
docker run --rm -p 7860:7860 eyeqc
```

The server has a light basic install. For optional vessel/FLAIR analysis, install `requirements-optional.txt`, upstream FLAIR (`pip install git+https://github.com/jusiro/FLAIR.git`), and allow its checkpoints and the RRWNet `j-morano/rrwnet-rite` checkpoint to be downloaded/cached. Confirm licenses and memory requirements first. Model loading is on demand and can have a substantial cold-start cost. If unavailable, the server returns a clear error rather than showing simulated scores.

**The published EyeQC hybrid model is not enabled:** its original DINOv2 preprocessing module was missing from the supplied release. Recover and verify it before turning on any risk scoring. The optional candidate encoder is an experimental reconstruction, not a validated clinical score.

## Run checks before every release

```bash
python scripts/check_pages.py
python -m pytest -q tests/
```

## Privacy and public redistribution

- Do not publish photographs without the right to redistribute them; the example data provided for this manuscript are not automatically public-domain.
- Never add patient names, identifiers or clinical reports to the example gallery.
- Run the backend behind HTTPS when using uploads; no authentication or long-term storage is included.
- The optional FLAIR/RRWNet outputs remain exploratory and must not be interpreted as validated EyeQC predictions.

For optional pretrained modules in the same container, use `docker build -f Dockerfile.full -t eyeqc-full .` and `docker run -p 7860:7860 eyeqc-full`. This larger variant installs PyTorch, RRWNet dependencies and the upstream FLAIR package. Cold-start downloads and model licenses still apply. This optional container configuration has not been verified with the upstream model downloads in an offline build.
