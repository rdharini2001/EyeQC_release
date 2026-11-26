"""Additional research tools that are NOT part of the paper or of the EyeQC model: RRWNet vessel analysis and FLAIR prompts."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:                                   # ZeroGPU on Hugging Face; a no-op elsewhere
    import spaces

    gpu = spaces.GPU
except ImportError:
    def gpu(*_, **__):
        return lambda function: function

import charts
from engine import read_photo

VESSEL_NOTE = ("RRWNet segmentation on a 576-pixel grid; measurements are in grid pixels, not micrometres. Not clinically validated and "
               "not used by EyeQC. Weights: j-morano/rrwnet-rite, CC BY-NC 4.0 (non-commercial).")
FLAIR_NOTE = ("FLAIR image-text similarity. Weights are relative to the prompts listed, so they are not disease probabilities or quality "
              "decisions; the maps are Grad-CAM saliency, not validated localisation. Not used by EyeQC.")


# Weights each tool needs.  They are fetched before the GPU call so that downloading them does not count against the
# short GPU time a ZeroGPU Space allows.
RRWNET_FILES = (("j-morano/rrwnet-rite", ("config.json", "model.safetensors")),)
FLAIR_FILES = (("jusiro2/FLAIR", ("config.json", "model.safetensors")),
               ("emilyalsentzer/Bio_ClinicalBERT", ("config.json", "vocab.txt", "model.safetensors")))


def _fetch(files) -> None:
    try:
        from huggingface_hub import snapshot_download

        for repository, names in files:
            snapshot_download(repository, allow_patterns=list(names))
    except Exception as problem:  # the tool itself then tries again and reports the error to the visitor
        print(f"Could not fetch weights in advance: {type(problem).__name__}: {problem}", flush=True)


def _device() -> str:
    return "cuda" if os.getenv("SPACES_ZERO_GPU", "").strip().lower() in ("1", "true", "yes") else "cpu"


def vessel_analysis(path: str):
    """(artery/vein overlay, vessel probability, centreline, measurements, note)."""
    _fetch(RRWNET_FILES)
    return _vessel_gpu(path)


@gpu(duration=60)
def _vessel_gpu(path: str):
    empty = pd.DataFrame(columns=["Measurement", "Value"])
    try:
        from eyeqc_vessel.engine import predict_vessels
        from eyeqc_vessel.morphometry import measure_morphometry
        from eyeqc_vessel.render import artery_vein_overlay, skeleton_overlay, vessel_probability

        rgb = read_photo(path)
        result = predict_vessels(rgb, device=_device())
        if result["degenerate"]:
            raise ValueError("no usable retinal field was found")
        measurements, _ = measure_morphometry(result["prob_canvas"][2], result["roi_canvas"],
                                              artery_probability=result["prob_canvas"][0], vein_probability=result["prob_canvas"][1])
        rows = []
        for name, value in measurements.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                rows.append({"Measurement": name, "Value": round(float(value), 3) if np.isfinite(value) else None})
            else:
                rows.append({"Measurement": name, "Value": str(value)})
        return artery_vein_overlay(rgb, result), vessel_probability(result), skeleton_overlay(rgb, result), pd.DataFrame(rows), VESSEL_NOTE
    except Exception as problem:
        return None, None, None, empty, f"The vessel model could not run: {type(problem).__name__}: {str(problem)[:300]}"


def flair_categories(path: str):
    """Relative weights of five fixed quality phrases: (chart, note)."""
    _fetch(FLAIR_FILES)
    return _flair_categories_gpu(path)


@gpu(duration=60)
def _flair_categories_gpu(path: str):
    try:
        from eyeqc_flair.engine import benchmark_image

        result = benchmark_image(read_photo(path), device=_device())
        figure = charts.weights(result["scores"], "category", "probability", "Five fixed quality phrases")
        return figure, f"{FLAIR_NOTE} Inference time {result['inference_ms'] / 1000:.1f} s."
    except Exception as problem:
        return charts.empty("No result"), f"FLAIR could not run: {type(problem).__name__}: {str(problem)[:300]}"


def flair_prompts(path: str, quality_prompt: str, pathology_prompt: str, more_prompts: str, explain: bool):
    """(weights chart, table, note, Grad-CAM for prompt 1, Grad-CAM for prompt 2, animation between the two)."""
    _fetch(FLAIR_FILES)
    return _flair_prompts_gpu(path, quality_prompt, pathology_prompt, more_prompts, explain)


@gpu(duration=75)
def _flair_prompts_gpu(path: str, quality_prompt: str, pathology_prompt: str, more_prompts: str, explain: bool):
    try:
        from eyeqc_flair.engine import compare_custom_prompts

        pair = [(quality_prompt or "").strip(), (pathology_prompt or "").strip()]
        if not all(pair):
            raise ValueError("Write the first two prompts.")
        prompts = pair + [line.strip() for line in (more_prompts or "").splitlines() if line.strip()]
        result = compare_custom_prompts(read_photo(path), prompts, device=_device(), explain_pair=bool(explain))
        table = pd.DataFrame(result["rows"])
        figure = charts.weights(result["rows"], "Prompt", "Relative weight", "Your prompts")
        maps = result["maps"]
        if maps is None:
            return figure, table, FLAIR_NOTE, None, None, None
        note = f"{FLAIR_NOTE} Overlap of the two saliency maps (intersection over union): {maps['saliency_overlap']:.2f}."
        return figure, table, note, maps["quality_overlay"], maps["pathology_overlay"], maps["animation"]
    except Exception as problem:
        empty = pd.DataFrame(columns=["Prompt", "Relative weight", "Similarity logit"])
        return charts.empty("No result"), empty, f"FLAIR could not run: {type(problem).__name__}: {str(problem)[:300]}", None, None, None
