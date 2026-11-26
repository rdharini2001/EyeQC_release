"""The released code reproduces the study's results on the bundled example photographs.

Needs the DINOv2 weights (downloaded on first use, about 330 MB).  Skip with:  pytest -m "not weights"
"""
import json
from pathlib import Path

import numpy as np
import pytest

from eyeqc.core import FEATURES

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [item for name in ("examples", "pair") for item in json.loads((ROOT / "app" / "examples" / f"{name}.json").read_text())]


@pytest.fixture(scope="module")
def model():
    from eyeqc.model import load_model

    return load_model()


def test_contract_shape():
    contract = json.loads((ROOT / "models" / "eyeqc_v3_model.json").read_text())
    assert contract["hand_features"] == list(FEATURES)
    assert len(contract["logistic_coefficients_standardized"]) == len(FEATURES) + 768


@pytest.mark.weights
@pytest.mark.parametrize("example", EXAMPLES, ids=[item["image_id"] for item in EXAMPLES])
def test_reproduces_study_result(model, example):
    from eyeqc.pipeline import read_rgb

    result = model.analyze(read_rgb(ROOT / "app" / "examples" / example["file"]))
    assert result["verdict"] == example["paper_verdict"]
    assert result["risk_bad"] == pytest.approx(example["paper_risk"], abs=0.01)
    total = result["base_calibrated_logit"] + sum(result["axis_contributions"].values()) + result["representation_contribution"]
    np.testing.assert_allclose(total, result["calibrated_logit"], atol=1e-6)
