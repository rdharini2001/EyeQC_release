"""Static "About" page: how the released model works, what each measurement is, and what the tool is not."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

from eyeqc.contract import FAIL_THRESHOLD, PASS_THRESHOLD
from eyeqc.core import ACTION_RULES, DISPLAY_NAMES, FEATURES

HERE = Path(__file__).resolve().parent
ASSETS = next((path for path in (HERE / "assets", HERE.parent / "assets") if path.is_dir()), HERE / "assets")


def _figure(name: str, alt: str) -> str:
    """An overview figure embedded as a compact WebP image."""
    buffer = io.BytesIO()
    Image.open(ASSETS / name).convert("RGBA").save(buffer, "WEBP", quality=92)
    return f"<figure class='eq-figure'><img src='data:image/webp;base64,{base64.b64encode(buffer.getvalue()).decode()}' alt='{html.escape(alt)}'/></figure>"


def landing_head() -> str:
    return ("<div class='eq-hero'><h1>EyeQC</h1><p>Retinal fundus image quality, explained: <b>what</b> is affected, <b>where</b>, and <b>what to do "
            "next</b>. A calibrated PASS / REVIEW / FAIL triage, ten named quality measurements, defect-specific maps and answers to your own "
            "quality questions.</p></div>")


def landing_figures() -> str:
    return (_figure("overview.png", "Routine review compared with the EyeQC workflow") +
            _figure("interface_overview.png", "The EyeQC interface and its research applications") +
            "<p class='eq-note center'>Figures are illustrative; the tool shows the values computed for your photograph.</p>")


DEFINITIONS = {
    "focus": "Variance of the Laplacian and strong gradients of the contrast-normalised green channel, penalised for noise.",
    "detail_uniformity": "Gradient detail in 6 × 6 retinal tiles, weighted towards the weakest tiles.",
    "local_readability": "Combined exposure, contrast and detail of the tiles, including the share of readable tiles.",
    "illumination_uniformity": "Irregularity of the low-frequency illumination left after fitting a smooth quadratic shading across the retina.",
    "exposure": "Median retinal brightness within a usable range, penalised for very dark or saturated pixels.",
    "contrast": "Spread between the 5th and 95th percentile of retinal brightness.",
    "field_coverage": "How much of the image, of its central region and of its extent the detected retinal field covers.",
    "artifact_burden": "Large specular glare and dark occlusions, and a penalty when little of the retina is readable.",
    "clipping": "Share of retinal colour values at the extremes of the range.",
    "color_balance": "Plausibility of the red-green and green-blue ratios and of colour saturation.",
}


def page() -> str:
    steps = [("Fundus photograph", "retinal field found"),
             ("10 quality measurements", "fixed image-processing rules, computed at ≤ 640 px"),
             ("DINOv2 ViT-B/14", "frozen encoder, 224 × 224 input, 768 features"),
             ("Logistic regression", "778 inputs, L2, Platt calibration"),
             ("Triage", f"PASS ≤ {PASS_THRESHOLD:.4f} · FAIL ≥ {FAIL_THRESHOLD:.4f} · REVIEW between")]
    flow = "<span class='eq-arrow'>→</span>".join(f"<div class='eq-step'><b>{name}</b><span>{text}</span></div>" for name, text in steps)
    rows = "".join(f"<tr><td><b>{DISPLAY_NAMES[name]}</b></td><td>{html.escape(DEFINITIONS[name])}</td><td>{html.escape(ACTION_RULES[name])}</td></tr>"
                   for name in FEATURES)
    return f"""
<div class='eq-card'><div class='eq-flow'>{flow}</div>
<p class='eq-note'>The measurements and the encoder run in parallel on the same photograph; their values are concatenated. Because the
predictor is linear, the calibrated log-odds split exactly into one contribution per measurement and one for the visual features.
These are model contributions, not causal effects.</p></div>
<div class='eq-card'><h3>The ten measurements</h3><table class='eq-table'><thead><tr><th>Measurement (0 to 1, 1 = best)</th><th>What it measures</th>
<th>Illustrative suggestion when low</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class='eq-card eq-two'><div><h3>Free-text requests</h3><p>A typed request is matched against a fixed vocabulary of quality terms (no language
model). It selects which measurements and maps are reported and composes the response from their values. It never changes the
risk score or the decision, and requests about disease or anatomy are declared not assessed.</p>
<h3>Defect-specific maps</h3><p>The exposure, contrast and detail rules evaluated in overlapping windows one eighth of the image wide. The combined
map is their geometric mean. They were evaluated on simulated defects only: exposure and detail maps find glare and shadow well,
the contrast and combined maps often do not, and no map explains the risk score.</p></div>
<div><h3>Intended use and limits</h3><ul><li>Technical quality of standard and portable colour fundus photographs. Not disease assessment.</li>
<li>Not for ultrawide-field imaging or autonomous decisions. Not a medical device.</li><li>Validation is retrospective; the reader study has
80 photographs.</li><li>Under distribution shift (DeepDRiD) poor-quality risk was under-estimated; recalibrate locally.</li>
<li>Next-step suggestions are illustrative and unvalidated.</li></ul>
<h3>Licences</h3><p>EyeQC code: MIT. DINOv2: Apache 2.0. Example photographs: DeepDRiD, CC BY-SA 4.0. Optional tools: RRWNet code MIT,
weights CC BY-NC 4.0 (non-commercial); FLAIR under its authors' terms.</p></div></div>
"""
