# Hugging Face Docker Space; build from the self-contained Space export.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    EYEQC_ENABLE_CANDIDATE_INFERENCE=0 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt requirements-optional.txt ./
# Load public RRWNet and FLAIR model implementations. The checkpoints are downloaded on demand.
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-optional.txt \
    && pip install --no-cache-dir git+https://github.com/jusiro/FLAIR.git
COPY --chown=user . .
USER user
RUN mkdir -p /home/user/.cache/huggingface /home/user/.cache/matplotlib
EXPOSE 7860
CMD ["uvicorn", "portal.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
