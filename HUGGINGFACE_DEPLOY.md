# Deploy the EyeQC tool to Hugging Face

**Research repository:** https://github.com/rdharini2001/EyeQC_release  
**Tool:** https://huggingface.co/spaces/Dharini24/EyeQC

The repository remains the reproducible research release. The Space is the only live web application. GitHub Pages is not required, and `.github/workflows/pages.yml` has been removed.

## Deploy to the existing Space

1. Back up any files you need from the existing `Dharini24/EyeQC` Space.
2. In the Space Settings or its README YAML, switch its SDK from `gradio` to **`docker`**. The bundled Space `README.md` already contains `sdk: docker` and `app_port: 7860`. The Space will rebuild when these files are committed.
3. From the research-repository root, export the standalone Space:

   ```bash
   python scripts/package_space.py --output ../EyeQC_hf_space
   ```

4. Upload the **contents** of `EyeQC_hf_space` to the Space repository (not a nested folder). One reliable method is:

   ```bash
   git clone https://huggingface.co/spaces/Dharini24/EyeQC eyeqc-space
   # Copy the exported files into eyeqc-space/ and remove obsolete old app files.
   cd eyeqc-space
   git add -A
   git commit -m "Launch unified EyeQC research workspace"
   git push
   ```

   Or upload the same files through the Hugging Face Space's Files tab. You need write access to the Space.

5. Once the Docker build finishes, open the Space and inspect `/health` and `/api/capabilities`. Both vessel and FLAIR buttons run within the page, with no external redirect.

## Models and first-run behavior

FLAIR code is installed from the public `jusiro/FLAIR` repository during the Docker build and its checkpoint is loaded from `jusiro2/FLAIR` when first requested. RRWNet code is included in this repository; its checkpoint comes from `j-morano/rrwnet-rite` when the vessel tool is first used. Both calls can take time to download weights and start on CPU. Enable Persistent Storage or suitable hardware if model reloads or timeouts become problematic.

The default Docker command uses one worker to avoid duplicated model memory. The trained EyeQC DINOv2 risk score is **not** enabled because the original encoder preprocessing parity remains unresolved. Do not override this limitation by enabling the experimental candidate in public deployment.

The code and dependencies may need adjustment if upstream libraries change. The downloadable release was tested for the local core API, but the Hugging Face container build and checkpoint downloads were **not** tested online.

## Privacy and source photographs

The app handles uploaded photographs in memory and does not save them to local disk. It is not a HIPAA-compliant or approved clinical service. Verify redistribution rights for the 11 sample photographs before making the Space public; replace unlicensed examples if necessary. No protected clinical metadata should be uploaded to the public demo.

## Check the release

```bash
pip install -r requirements.txt pytest
python -m pytest -q tests/
python scripts/check_pages.py
```

For a local full-container preview:

```bash
python scripts/package_space.py --output /tmp/eyeqc-space-export
docker build -t eyeqc-space /tmp/eyeqc-space-export
docker run --rm -p 7860:7860 eyeqc-space
# http://localhost:7860
```
