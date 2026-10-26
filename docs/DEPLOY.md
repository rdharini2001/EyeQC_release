# The EyeQC website is hosted on Hugging Face

The research website in this `docs/` directory is served by the Python FastAPI application in `portal/`, both locally and on Hugging Face Spaces. GitHub Pages is no longer the deployment target.

**Live website:** https://huggingface.co/spaces/Dharini24/EyeQC  
**Research repository:** https://github.com/rdharini2001/EyeQC_release

For deployment and the standalone Space packaging command, read [`HUGGINGFACE_DEPLOY.md`](../HUGGINGFACE_DEPLOY.md).

The Python service provides ten-feature extraction, controlled degradations and dose curves for uploaded photographs, batch exports, optional RRWNet artery/vein analysis, and optional FLAIR image–text comparisons. FLAIR and RRWNet checkpoints are downloaded separately from their public model repositories. The exact published EyeQC risk score remains disabled until DINOv2 encoder preprocessing parity is established.
