"""Location of the frozen model file and its operating thresholds (no deep-learning dependency)."""
from __future__ import annotations

import json
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "eyeqc_v3_model.json"
_CONTRACT = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
PASS_THRESHOLD = float(_CONTRACT["pass_risk_threshold"])   # risk <= this: PASS
FAIL_THRESHOLD = float(_CONTRACT["fail_risk_threshold"])   # risk >= this: FAIL; in between: REVIEW
