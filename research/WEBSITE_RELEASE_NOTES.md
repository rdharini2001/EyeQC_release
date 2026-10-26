# Website and research release

The EyeQC research repository and the live EyeQC tool have separate roles.

- GitHub (`rdharini2001/EyeQC_release`) holds the quality extractor, experiment scripts, selected aggregate validation results, source photographs where redistributable, model contract, documentation and research figures.
- Hugging Face (`Dharini24/EyeQC`) hosts the working research application on a Docker Space. The front end and Python API run together on one host; the optional RRWNet vessel model and FLAIR comparison do not redirect to other websites.

See `HUGGINGFACE_DEPLOY.md` and run `python scripts/package_space.py --output ../EyeQC_hf_space` from the repository root to prepare a deployable Space.

Known limitation: the validated calibrated EyeQC risk head cannot be used on new photographs until the original DINOv2 embedding preprocessing is verified. The spatial maps shown today are code-computed tile readability measurements rather than a direct contribution map of the 778-feature classifier.
