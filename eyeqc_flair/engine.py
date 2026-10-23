"""Zero-shot benchmarking against FLAIR (Foundation Language-Image model of
the Retina), a public vision-language model pretrained on fundus photographs
(https://github.com/jusiro/FLAIR, weights hosted at HF Hub ``jusiro2/FLAIR``).

This is a secondary, research-facing "benchmarking" feature, not part of the
primary clinical flow: it shows how a general-purpose foundation model reads
the same photograph EyeQC's own trained model just scored, for comparison.
It has no bearing on EyeQC's own verdict.

Loading follows the same lazy-singleton-on-first-use pattern as the app's
own DINOv2 backbone (see ``eyeqc_v2/backbone.py``): nothing is downloaded or
loaded until the first real call, so it never taxes app startup, and after
that first call the process-wide singleton is reused.

Prompting follows FLAIR's own documented convention (see
``flair.modeling.model.FLAIRModel``): a fixed caption template,
``"A fundus photograph of [CLS]"``, with ``[CLS]`` substituted by each
candidate category phrase, then compared to the image via cosine similarity
in FLAIR's joint embedding space. Categories below are noun phrases so the
filled-in template reads naturally (mirroring how the model's own
pretraining used diagnostic categories such as "diabetic retinopathy" in the
same slot) and are framed around image *quality*, since that — not disease
diagnosis — is this app's domain.
"""
from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

# (internal_key, phrase that fills the "[CLS]" slot, user-facing label)
CATEGORY_PROMPTS: list[tuple[str, str, str]] = [
    (
        "quality_ok",
        "sufficient technical quality for clinical reading, sharp focus and clear detail",
        "Good technical quality",
    ),
    (
        "blur",
        "blur or poor focus",
        "Blur or focus concern",
    ),
    (
        "dark",
        "an image too dark to read, with insufficient illumination",
        "Underexposed / too dark",
    ),
    (
        "glare",
        "glare or overexposed, washed-out regions",
        "Glare or overexposure",
    ),
    (
        "artifact",
        "artifacts or debris obscuring the retina",
        "Artifacts obscuring the image",
    ),
]
_PHRASES = [phrase for _, phrase, _ in CATEGORY_PROMPTS]
_KEY_TO_LABEL = {key: label for key, _, label in CATEGORY_PROMPTS}

HF_HUB_REPO = "jusiro2/FLAIR"

_MODEL_CACHE: dict[str, object] = {}


class FlairUnavailableError(RuntimeError):
    """Raised when FLAIR's weights can't be loaded (no network, HF Hub
    unreachable, incompatible/missing install, etc). Callers (the FastAPI
    endpoint) should turn this into a 503, never a crash."""


def load_flair(device: str = "cpu"):
    """Load the pretrained FLAIR zero-shot model once per process.

    Mirrors ``eyeqc_v2.backbone.load_backbone``'s lazy-singleton pattern.
    Deliberately NOT called from the app's startup warm-up — this is an
    optional, secondary feature and should only pay its (first-call, network
    + ~1.3GB weight download) cost when someone actually requests it.
    """
    key = f"flair:{device}"
    if key not in _MODEL_CACHE:
        try:
            from flair import FLAIRModel
        except Exception as exc:  # noqa: BLE001 - package missing/broken
            raise FlairUnavailableError(
                f"The FLAIR package is not installed or failed to import: {exc}"
            ) from exc

        try:
            model = FLAIRModel.from_pretrained(HF_HUB_REPO)
        except Exception as exc:  # noqa: BLE001 - network/HF Hub/torch errors
            raise FlairUnavailableError(
                f"Could not download or load FLAIR's pretrained weights from "
                f"Hugging Face Hub ({HF_HUB_REPO}): {exc}"
            ) from exc

        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


class FlairBenchmarkResult(NamedTuple):
    predicted_key: str
    predicted_label: str
    confidence: float
    scores: list[dict]
    inference_ms: float
    cold_start: bool
    checkpoint: str


def benchmark_image(rgb: np.ndarray, *, device: str = "cpu") -> dict:
    """Run FLAIR's zero-shot quality read on one in-memory RGB image.

    Returns predicted category + confidence, the raw per-prompt similarity
    scores, and how long the forward pass alone took (excluding any one-time
    model load, so the timing is apples-to-apples with EyeQC's own
    steady-state ``runtime_end_to_end_sec``).
    """
    cold_start = f"flair:{device}" not in _MODEL_CACHE
    model = load_flair(device=device)

    t0 = time.perf_counter()
    try:
        probs, logits = model(rgb, _PHRASES)
    except Exception as exc:  # noqa: BLE001 - malformed image, OOM, etc
        raise FlairUnavailableError(f"FLAIR inference failed: {exc}") from exc
    inference_ms = (time.perf_counter() - t0) * 1000.0

    probs_row = np.asarray(probs)[0]
    logits_row = np.asarray(logits)[0]
    top_index = int(np.argmax(probs_row))
    predicted_key, _, predicted_label = CATEGORY_PROMPTS[top_index]

    scores = [
        {
            "category": label,
            "probability": round(float(probs_row[i]), 4),
            "raw_similarity": round(float(logits_row[i]), 4),
        }
        for i, (_, _, label) in enumerate(CATEGORY_PROMPTS)
    ]
    scores.sort(key=lambda row: row["probability"], reverse=True)

    return FlairBenchmarkResult(
        predicted_key=predicted_key,
        predicted_label=predicted_label,
        confidence=round(float(probs_row[top_index]), 4),
        scores=scores,
        inference_ms=round(inference_ms, 1),
        cold_start=cold_start,
        checkpoint=HF_HUB_REPO,
    )._asdict()
