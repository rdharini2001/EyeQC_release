"""Answer a free-text quality request from the measurements of one photograph.

The request is interpreted by :mod:`eyeqc.criteria` (fixed vocabulary, no language model).  The answer is assembled from
templates using only the requested measurements, their development reference values and the defect-specific maps.  It
never changes the risk score or the decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ACTION_RULES, DISPLAY_NAMES
from .criteria import VOCABULARY, Criteria
from .maps import LOCALISING_KEYS
from .reference import GRADABLE_P10, is_low


@dataclass(frozen=True)
class Finding:
    name: str
    label: str
    value: float
    reference: float
    low: bool
    map_key: str | None             # a map that localises this property reasonably well, if any
    low_area: float | None          # share of the retina scoring below 0.5 in that map
    advice: str                     # acquisition suggestion, only when the measurement is low


@dataclass(frozen=True)
class Answer:
    headline: str
    findings: tuple[Finding, ...]
    decision: str
    not_assessed: tuple[str, ...]


def _low_area(maps: dict | None, key: str | None, inner: np.ndarray | None) -> float | None:
    if maps is None or key is None or inner is None:
        return None
    score = maps[key][inner]
    score = score[np.isfinite(score)]
    return float((score < 0.5).mean()) if score.size else None


def answer(criteria: Criteria, result: dict, maps: dict | None = None) -> Answer:
    """``result`` is the output of ``EyeQCModelV3.analyze``; ``maps`` the output of ``quality_maps``."""
    features = result["features"]
    findings = []
    for name in criteria.measurements:
        value = float(features[name])
        key = next((k for k in VOCABULARY[name][1] if k in LOCALISING_KEYS), None)
        low = is_low(name, value)
        findings.append(Finding(name=name, label=DISPLAY_NAMES[name], value=value, reference=GRADABLE_P10[name], low=low, map_key=key,
                                low_area=_low_area(maps, key, features.get("inner_mask")), advice=ACTION_RULES[name] if low else ""))
    findings.sort(key=lambda finding: not finding.low)      # measurements below reference first, otherwise report order
    low = [finding.label.lower() for finding in findings if finding.low]
    count = len(findings)
    if not count:
        headline = "No measurement was requested."
    elif not low:
        headline = (f"{findings[0].label} is at or above the development reference." if count == 1 else
                    f"None of the {count} requested measurements is below the development reference.")
    elif count == 1:
        headline = f"{findings[0].label} is below the development reference."
    else:
        listed = ", ".join(low[:-1]) + (" and " if len(low) > 1 else "") + low[-1]
        headline = f"{len(low)} of {count} requested measurements {'is' if len(low) == 1 else 'are'} below the development reference: {listed}."
    decision = (f"Overall decision: {result['verdict']} (poor-quality risk {result['risk_bad']:.2f}). It uses all ten measurements and the "
                "visual features, so it can differ from what the requested measurements suggest.")
    return Answer(headline=headline, findings=tuple(findings), decision=decision, not_assessed=criteria.out_of_scope)
