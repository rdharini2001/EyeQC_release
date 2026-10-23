# Publish the EyeQC research website through GitHub Pages

This repository is ready for GitHub Pages. You do **not** need to run the Python backend to view the website.

## Option A: Directly from the `/docs` folder (easiest)

1. Create or open the intended GitHub repository (for example `rdharini2001/EyeQC`). Make sure it is available under the account where you want to publish the site.
2. Upload **the contents of this repository ZIP** to its root. The `docs/` directory must be at the root beside `README.md`.
3. Commit everything to the `main` branch. In a terminal:

   ```bash
   git add .
   git commit -m "Build EyeQC research website and reproducible image gallery"
   git push origin main
   ```

4. Open **Settings → Pages** on GitHub.
5. Under **Build and deployment**, select **Deploy from a branch**. Choose `main` and `/docs`, then **Save**.
6. The expected address for a repository named `EyeQC` under `rdharini2001` is:
   `https://rdharini2001.github.io/EyeQC/`
7. After the initial build, try the page in an incognito window or force-refresh if an older page is cached.

### Option B: GitHub Actions

The repository also includes `.github/workflows/pages.yml`. It checks the real image/maps/data files and publishes `docs/` on pushes to `main`.

In **Settings → Pages → Build and deployment**, choose **GitHub Actions**. Commit and push, then use the **Actions** tab to see the job finish. GitHub Actions itself handles the publish; do not also select branch source.

## What GitHub Pages can and cannot do

Works directly: all sample photographs and measured maps, overlays and download buttons, nine-defect dose lab and 5-point charts, study/baseline charts, and browser-only upload preview.

Needs a Python server somewhere else: calibrated DINOv2+EyeQC scoring, true ten-feature extraction on arbitrary new uploads, and optional FLAIR and vessel pretrained networks. The packaged `portal/app.py` is available for that deployment, but trained-risk inference needs validation of the original encoder preprocessing before enabling.

For a local preview: `python -m http.server 8000 --directory docs`.

## Before making the repository public

- Confirm permission to redistribute the example photographs; replace or delete those without permission.
- Check your own code and model licenses, especially third-party pretrained weights.
- Add approved ethics, publication, and data use terms.
- Do not commit original patient-level files, model-training cache or private records.
- Check README links for your repository owner and chosen repository name.

## If the old website is still showing

The old Hugging Face Space is a separate service. Pushing a new repository to GitHub **does not update a Hugging Face Space**. The new GitHub Pages site will have the `github.io` address above. An existing `github.io` Pages URL may also remain old until Pages is reconfigured to point to this repo's `main/docs` or Actions artifact. The older Space can remain as a separate research backend or be replaced later.
