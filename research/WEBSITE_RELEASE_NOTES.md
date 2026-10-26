# EyeQC unified website release

## What changed

The old public site showed the quality explorer but hid the vessel and FLAIR tools behind the optional API panel. The release makes these tools visible directly in the main research site, while retaining the same interface for static and Python hosting.

The code now provides:

- Real retinal photograph workbench with ten interpretable measurements and 6×6 measured tile maps.
- Code-generated vessel contrast gallery computed with OpenCV; the RRWNet inference button is separate and runs only with its upstream weights installed.
- A dedicated FLAIR research panel returning model prompt similarities when the FLAIR package and weights are available.
- Nine controlled acquisition defects, five doses, saved altered photographs and dose-response plots from the original research implementation.
- Two new figure atlases rebuilt from the supplied retinal images and measurements.
- Same-origin Python API serving the identical site; no redirection to the previous Hugging Face interface.
- A fixed backend startup issue (`portal/static` was absent from the previous release) and corrected code links pointing to `EyeQC_release`.

## Known limitations

GitHub Pages cannot execute PyTorch models. Static sample views work without a server, while RRWNet, FLAIR and Python processing on newly uploaded images require the optional backend. The EyeQC hybrid classifier cannot be represented as validated inference until the original DINOv2 preprocessing contract is restored and checked against archived outputs. FLAIR and RRWNet are additional research modules, not the studied model.

The display contours are visualization aids, not validated clinical operating thresholds. Example photographs must be cleared for public redistribution by the dataset rights holder.

## To publish

Download the full ZIP, extract its contents at repository root, review the image rights, and commit to `main`. Configure GitHub Pages to serve `main/docs` or use the supplied Actions workflow. The public site will then be `https://rdharini2001.github.io/EyeQC_release/`.
