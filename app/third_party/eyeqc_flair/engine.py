"""Optional zero-shot comparison with FLAIR, a vision-language model pretrained on retinal photographs
(https://github.com/jusiro/FLAIR, weights ``jusiro2/FLAIR``).  Nothing here changes EyeQC's score or decision.

The model is loaded on the first request.  Prompt weights are a softmax over the prompts supplied, so they depend on that
list; they are not probabilities of disease or of adequate quality.
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
    """FLAIR could not be loaded or run."""


def load_flair(device: str = "cpu"):
    """Load FLAIR once per process on ``cpu`` or ``cuda`` (first use downloads about 1.3 GB)."""
    if device not in ("cpu", "cuda"):
        raise FlairUnavailableError(f"Unsupported FLAIR device: {device!r}")
    key = f"flair:{device}"
    if key not in _MODEL_CACHE:
        try:
            # FLAIR uses its own module-global `device`. Set this explicitly
            # before constructing the model. The 'cuda' path is only invoked
            # from a Gradio callback wrapped with @spaces.GPU, i.e. inside
            # the real ZeroGPU worker. This avoids the low-level CUDA init
            # error in the ZeroGPU *main process*.
            from flair import FLAIRModel
            from flair.modeling import model as upstream_flair_model
            if not hasattr(upstream_flair_model, "device"):
                raise RuntimeError(
                    "FLAIR upstream device API changed; review required"
                )
            upstream_flair_model.device = device
        except Exception as exc:  # noqa: BLE001 - package missing/broken
            raise FlairUnavailableError(
                f"The FLAIR package is not installed or failed to import: {exc}"
            ) from exc

        try:
            # Loading on CPU avoids GPU device-map deserialization
            # and low-level CUDA init outside supported torch entry points.
            # The model's own .to(device) runs inside the allocated GPU worker
            # on ZeroGPU. On CPU Basic it remains entirely on CPU.
            model = FLAIRModel.from_pretrained(HF_HUB_REPO, map_location="cpu")
            model = model.to(device)
        except Exception as exc:  # noqa: BLE001 - network/HF Hub/torch errors
            raise FlairUnavailableError(
                f"FLAIR model initialization failed for device={device} "
                f"with checkpoint {HF_HUB_REPO}: {type(exc).__name__}: {exc}"
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
    """Weights of five fixed quality phrases for one RGB photograph."""
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



def compare_custom_prompts(rgb: np.ndarray, prompts: list[str], *, device: str="cpu", explain_pair: bool=False) -> dict:
    """Compare user-authored FLAIR prompts with optional *real* ResNet Grad-CAM.

    Relative weights are softmax within the *supplied list*, and are not
    probabilities that image quality is sufficient or that disease is present.
    """
    phrases=[str(p).strip() for p in prompts if str(p).strip()]
    if not 2<=len(phrases)<=8:
        raise ValueError("Enter 2–8 nonempty prompts, including a quality and pathology prompt")
    if any(len(s)>180 for s in phrases):
        raise ValueError('Each text prompt must be 180 characters or fewer')
    model=load_flair(device)
    probs,logits=model(rgb,phrases)
    rows=[{'Prompt':s,'Relative weight':round(float(probs[0,i]),4),
           'Similarity logit':round(float(logits[0,i]),3)} for i,s in enumerate(phrases)]
    output={'rows':rows,'maps':None,'checkpoint':HF_HUB_REPO}
    if explain_pair:
        from .explanations import explain_flair_gradcam
        output['maps']=explain_flair_gradcam(model,rgb,phrases[0],phrases[1])
    return output
