"""The EyeQC model used in the paper (version 3).

Ten predefined quality measurements are concatenated with a frozen DINOv2 embedding (778 values), standardised, and
passed to one logistic regression with Platt calibration.  All numbers are read from ``models/eyeqc_v3_model.json``.

    from eyeqc.model import load_model
    from eyeqc.pipeline import read_rgb

    result = load_model().analyze(read_rgb("photo.jpg"))
    result["verdict"], result["risk_bad"]
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .backbone import embed_rgb
from .contract import MODEL_PATH
from .core import ACTION_RULES, FEATURES, extract_features


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


class EyeQCModelV3:
    """Inference-only, contract-frozen hybrid model."""

    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        self.contract = json.loads(self.path.read_text(encoding="utf-8"))
        self.mean = np.asarray(self.contract["standardizer_mean"], float)
        self.scale = np.asarray(self.contract["standardizer_scale"], float)
        self.coef = np.asarray(self.contract["logistic_coefficients_standardized"], float)
        self.intercept = float(self.contract["logistic_intercept"])
        if np.any(self.scale <= 0):
            raise ValueError("standardizer scales must be positive")
        pass_t = float(self.contract["pass_risk_threshold"])
        fail_t = float(self.contract["fail_risk_threshold"])
        if not 0 <= pass_t < fail_t <= 1:
            raise ValueError("invalid PASS/FAIL risk thresholds")

    def score(self, hand_features: dict, embedding: np.ndarray) -> dict:
        x = np.concatenate([np.asarray([hand_features[name] for name in FEATURES], float), embedding])
        z = (x - self.mean) / self.scale
        raw_logit = float(self.intercept + z @ self.coef)
        calibrated_logit = float(self.contract["platt_intercept"] + self.contract["platt_coefficient"] * raw_logit)
        # Exact decomposition of the calibrated predictor about its frozen
        # standardized-input origin. These terms are model attributions,
        # not effects of independent interventions on the photograph.
        calibrated_terms = float(self.contract["platt_coefficient"]) * z * self.coef
        base_logit = float(self.contract["platt_intercept"] + self.contract["platt_coefficient"] * self.intercept)
        risk = _sigmoid(calibrated_logit)
        pass_t = float(self.contract["pass_risk_threshold"])
        fail_t = float(self.contract["fail_risk_threshold"])
        verdict = "FAIL" if risk >= fail_t else "PASS" if risk <= pass_t else "REVIEW"
        weakest = min(FEATURES, key=lambda k: hand_features[k])
        return {
            "model_version": self.contract["version"],
            "feature_hash": self.contract.get("development_manifest_sha256", "")[:12],
            "risk_bad": risk,
            "quality_score": 1 - risk,
            "verdict": verdict,
            "weakest_axis": weakest,
            "suggested_action": ACTION_RULES[weakest],
            "raw_logit": raw_logit,
            "calibrated_logit": calibrated_logit,
            "base_calibrated_logit": base_logit,
            "axis_contributions": {name: float(calibrated_terms[i]) for i, name in enumerate(FEATURES)},
            "representation_contribution": float(calibrated_terms[len(FEATURES):].sum()),
        }

    def analyze(self, rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        features = extract_features(rgb)
        embedding = embed_rgb(rgb)
        scored = self.score(features, embedding)
        scored["features"] = features
        scored["runtime_end_to_end_sec"] = time.perf_counter() - t0
        return scored


def load_model(path: str | Path = MODEL_PATH) -> EyeQCModelV3:
    """Load the frozen model released with the paper."""
    return EyeQCModelV3(path)
