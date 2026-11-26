"""Reference values from the development cohort, used only to highlight low measurements in the interface and in cohort summaries.

Each value is the 10th percentile of that measurement among the 10223 gradable development photographs (EyeQ training cohort).
Highlighting does not change the risk score or the PASS / REVIEW / FAIL decision.
"""
from __future__ import annotations

GRADABLE_P10: dict[str, float] = {
    "focus": 1.0000,
    "detail_uniformity": 0.9964,
    "local_readability": 0.2746,
    "illumination_uniformity": 0.7358,
    "exposure": 0.8961,
    "contrast": 0.2375,
    "field_coverage": 0.6922,
    "artifact_burden": 0.8133,
    "clipping": 0.8200,
    "color_balance": 1.0000,
}
TOLERANCE = 1e-3     # focus and color balance sit at 1.0 for most photographs; ignore differences smaller than this


def is_low(name: str, value: float) -> bool:
    """Whether a measurement is below the development reference."""
    return value < GRADABLE_P10[name] - TOLERANCE
