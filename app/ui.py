"""HTML / SVG rendering of the EyeQC window: quality report, triage gauge, defect-specific heatmaps and score decomposition."""
from __future__ import annotations

import base64
import html
import math

import cv2
import numpy as np

from eyeqc.contract import FAIL_THRESHOLD, PASS_THRESHOLD
from eyeqc.core import DISPLAY_NAMES, FEATURES
from eyeqc.criteria import Criteria
from eyeqc.maps import MAP_KEYS
from eyeqc.reference import is_low

from engine import Assessment, crop as _crop

MAP_TITLES = {"exposure": "Exposure", "contrast": "Contrast", "detail": "Detail", "quality": "EyeQC quality map"}
GREEN, AMBER, RED = "#5DB386", "#FBC82B", "#F4686C"


def esc(text) -> str:
    return html.escape(str(text))


def data_uri(rgba: np.ndarray) -> str:
    code = cv2.COLOR_RGBA2BGRA if rgba.shape[2] == 4 else cv2.COLOR_RGB2BGR
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, code))
    if not ok:
        raise ValueError("could not encode image")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


# ---------------------------------------------------------------- quality report
def gauge_angle(risk: float) -> float:
    """Needle angle in degrees (180 = left end, 360 = right end).

    The three arcs have equal-looking widths for legibility; the needle is placed proportionally *within* the arc of its
    category, so it always points into the arc that matches the decision.
    """
    if risk <= PASS_THRESHOLD:
        return 180 + 58 * risk / PASS_THRESHOLD
    if risk < FAIL_THRESHOLD:
        return 240 + 60 * (risk - PASS_THRESHOLD) / (FAIL_THRESHOLD - PASS_THRESHOLD)
    return 302 + 58 * (risk - FAIL_THRESHOLD) / (1 - FAIL_THRESHOLD)


def _arc(start: float, end: float, colour: str, radius: float = 92, width: float = 24) -> str:
    cx, cy = 130, 112
    points = lambda r, a: (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
    (x0, y0), (x1, y1) = points(radius, start), points(radius, end)
    (x2, y2), (x3, y3) = points(radius - width, end), points(radius - width, start)
    return (f"<path d='M{x0:.1f},{y0:.1f} A{radius},{radius} 0 0 1 {x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f} "
            f"A{radius - width},{radius - width} 0 0 0 {x3:.1f},{y3:.1f} Z' fill='{colour}'/>")


def gauge(risk: float | None) -> str:
    parts = [_arc(180, 239, GREEN), _arc(241, 299, AMBER), _arc(301, 360, RED)]
    for angle, value in ((240, PASS_THRESHOLD), (300, FAIL_THRESHOLD)):
        x, y = 130 + 104 * math.cos(math.radians(angle)), 112 + 104 * math.sin(math.radians(angle))
        parts.append(f"<text x='{x:.1f}' y='{y:.1f}' class='eq-tick' text-anchor='middle'>{value:.2f}</text>")
    if risk is not None:
        a = math.radians(gauge_angle(risk))
        tip = (130 + 78 * math.cos(a), 112 + 78 * math.sin(a))
        left = (130 + 6 * math.cos(a + math.pi / 2), 112 + 6 * math.sin(a + math.pi / 2))
        right = (130 + 6 * math.cos(a - math.pi / 2), 112 + 6 * math.sin(a - math.pi / 2))
        parts.append(f"<polygon points='{left[0]:.1f},{left[1]:.1f} {tip[0]:.1f},{tip[1]:.1f} {right[0]:.1f},{right[1]:.1f}' fill='#F4F7F8'/>")
    parts.append("<circle cx='130' cy='112' r='9' fill='#F4F7F8'/>")
    return f"<svg viewBox='0 0 260 126' class='eq-gauge' role='img' aria-label='triage gauge'>{''.join(parts)}</svg>"


def report(assessment: Assessment | None, show_values: bool = False) -> str:
    rows = []
    for name in FEATURES:
        if assessment is None:
            value, colour, dim, shown = 0.0, "", "", ""
        else:
            value = float(assessment.features[name])
            colour = " low" if is_low(name, value) else ""
            dim = "" if name in assessment.criteria.measurements else " dim"
            shown = f"{value:.2f}"
        number = f"<span class='eq-num'>{shown}</span>" if show_values else ""
        rows.append(f"<div class='eq-row{dim}' title='{esc(DISPLAY_NAMES[name])}: {shown}'><span class='eq-name'>{esc(DISPLAY_NAMES[name])}</span>"
                    f"<span class='eq-track'><span class='eq-fill{colour}' style='width:{100 * value:.0f}%'></span></span>{number}</div>")
    verdict = assessment.result["verdict"] if assessment else None
    risk = assessment.result["risk_bad"] if assessment else None
    buttons = "".join(f"<span class='eq-verdict {v.lower()}{' on' if v == verdict else ''}'>{v}</span>" for v in ("PASS", "REVIEW", "FAIL"))
    if assessment is None:
        footer = "<div class='eq-risk'>Upload a photograph and press <b>Assess image</b>.</div>"
    else:
        headline, detail = assessment.next_step()
        scope = "" if assessment.in_scope else "<div class='eq-warn'>This input may not be a colour fundus photograph; interpret with caution.</div>"
        footer = (f"<div class='eq-risk'>Poor-quality risk <b>{risk:.2f}</b><i>scored in {assessment.seconds:.1f} s</i></div>{scope}"
                  f"<div class='eq-next {verdict.lower()}'><b>Next step (illustrative) · {headline}</b><span>{esc(detail)}</span></div>")
    return (f"<div class='eq-panel'><div class='eq-title'>Quality report</div><div class='eq-bars'>{''.join(rows)}</div>"
            f"<div class='eq-legend'><i></i>at or above development reference <i class='low'></i>below it</div>"
            f"<div class='eq-rule'></div><div class='eq-title'>Triage</div>{gauge(risk)}<div class='eq-verdicts'>{buttons}</div>{footer}</div>")


# ---------------------------------------------------------------- heatmaps
def _map_cell(tile, key: str, dimmed: bool, show_star: bool) -> str:
    shapes = []
    if tile.outline:
        for polygon in tile.outline:
            points = " ".join(f"{100 * x:.1f},{100 * y:.1f}" for x, y in polygon)
            shapes.append(f"<polygon points='{points}' class='eq-outline'/>")
    if show_star and tile.star:
        x, y = 100 * tile.star[0], 100 * tile.star[1]
        star = " ".join(f"{x + r * math.cos(math.radians(-90 + 36 * i)):.2f},{y + r * math.sin(math.radians(-90 + 36 * i)):.2f}"
                        for i, r in zip(range(10), [3.6, 1.5] * 5))
        shapes.append(f"<polygon points='{star}' class='eq-star'/>")
    classes = "eq-cell" + (" dim" if dimmed else "")
    value_class = "eq-value combined" if key == "quality" else "eq-value"
    return (f"<div class='{classes}'><div class='eq-map'><img src='{data_uri(tile.image)}' alt='{key} map'/>"
            f"<svg viewBox='0 0 100 100'>{''.join(shapes)}</svg></div><div class='{value_class}'>{tile.value}</div></div>")


def heatmaps(assessment: Assessment | None, style: str = "spectral") -> str:
    header = "".join(f"<div class='eq-col {key}'>{MAP_TITLES[key]}</div>" for key in MAP_KEYS)
    body = ""
    if assessment is None:
        body = "<div class='eq-empty'>Maps appear here after assessment.</div>"
    else:
        for label, tiles in assessment.rows.items():
            cells = "".join(_map_cell(tiles[key], key, key not in assessment.criteria.maps, True) for key in MAP_KEYS)
            body += f"<div class='eq-maprow'><div class='eq-pill'>{esc(label)}</div>{cells}</div>"
    scale = ("<span class='eq-ramp spectral'></span>" if style == "spectral" else "<span class='eq-ramp drop'></span>")
    ends = ("poor", "good") if style == "spectral" else ("small drop", "large drop")
    demo = assessment is not None and len(assessment.rows) > 1
    note = ("<b>This photo:</b> share of the retina scoring below 0.5. <b>Glare, Shadow:</b> the same photograph with a simulated "
            "defect (dashed outline); number = pixel AUC for finding it (0.5 = chance). Exposure and detail maps outline defects "
            "best; contrast and combined maps often do not." if demo else
            "Number = share of the retina scoring below 0.5.")
    return (f"<div class='eq-panel'><div class='eq-title'>Defect-specific heatmaps</div><div class='eq-rule'></div>"
            f"<div class='eq-maphead'><div></div>{header}</div>{body}"
            f"<div class='eq-scale'><span>{ends[0]}</span>{scale}<span>{ends[1]}</span><span class='eq-star-key'>★ lowest-scoring region</span></div>"
            f"<div class='eq-note'>{note}</div></div>")


# ---------------------------------------------------------------- explanation under the window
def decomposition(assessment: Assessment | None) -> str:
    if assessment is None:
        return ""
    result = assessment.result
    terms = [("Intercept", result["base_calibrated_logit"])]
    terms += [(DISPLAY_NAMES[name], result["axis_contributions"][name]) for name in FEATURES]
    terms.append(("Visual features (DINOv2)", result["representation_contribution"]))
    largest = max(abs(value) for _, value in terms) or 1.0
    rows = ""
    for name, value in terms:
        width = 50 * abs(value) / largest
        left = 50 if value >= 0 else 50 - width
        rows += (f"<div class='eq-term'><span class='eq-name'>{esc(name)}</span><span class='eq-axis'>"
                 f"<span class='eq-bar {'up' if value >= 0 else 'down'}' style='left:{left:.1f}%;width:{width:.1f}%'></span></span>"
                 f"<span class='eq-num'>{value:+.2f}</span></div>")
    total = result["calibrated_logit"]
    return (f"<div class='eq-panel'><div class='eq-note'>The calibrated log-odds of poor quality is the sum of these terms. "
            f"Red raises the risk, teal lowers it. These are model contributions, not causal effects.</div>{rows}"
            f"<div class='eq-total'>Sum {total:+.2f} → risk {result['risk_bad']:.3f} "
            f"(PASS ≤ {PASS_THRESHOLD:.4f}, FAIL ≥ {FAIL_THRESHOLD:.4f}, REVIEW between)</div></div>")


def summary(assessment: Assessment | None) -> str:
    if assessment is None:
        return ""
    return (f"<div class='eq-summary'><b>Criteria.</b> {esc(assessment.criteria.summary())} "
            f"The risk score and decision always use all ten measurements and the visual features.</div>")


# ---------------------------------------------------------------- page furniture
def brand() -> str:
    dots = "".join(f"<span class='eq-dot' style='background:{colour}'></span>" for colour in ("#F25F57", "#FDBC40", "#33C748"))
    return (f"<div class='eq-brand'><span class='eq-dots'>{dots}</span>"
            "<svg width='34' height='22' viewBox='0 0 34 22'><path d='M2 11C8 2.5 26 2.5 32 11 26 19.5 8 19.5 2 11Z' fill='none' stroke='#6FE0D6' "
            "stroke-width='2.4'/><circle cx='17' cy='11' r='4.6' fill='#6FE0D6'/></svg>EyeQC</div>")


def section(title: str, text: str, in_paper: bool) -> str:
    badge = "<em class='eq-badge'>In the paper</em>" if in_paper else "<em class='eq-badge other'>Not in the paper</em>"
    return f"<div class='eq-section'><h2>{esc(title)}{badge}</h2><p>{esc(text)}</p></div>"


def localisation(photo, tiles: dict, truth, box, result: dict, defect: str) -> str:
    """The degraded photograph with the defect outline, followed by the four maps."""
    contours, _ = cv2.findContours(_crop(truth.astype(np.uint8), box), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    side = box[1] - box[0]
    shapes = "".join("<polygon points='" + " ".join(f"{100 * x / side:.1f},{100 * y / side:.1f}" for x, y in c[:, 0, :]) + "' class='eq-outline'/>"
                     for c in contours if len(c) > 2)
    cells = (f"<div class='eq-cell'><div class='eq-col'>{esc(defect)} added</div><div class='eq-map'><img src='{data_uri(photo)}' alt='degraded photograph'/>"
             f"<svg viewBox='0 0 100 100'>{shapes}</svg></div><div class='eq-value'>{result['verdict']} · risk {result['risk_bad']:.2f}</div></div>")
    for key in MAP_KEYS:
        cells += (f"<div class='eq-cell'><div class='eq-col {key}'>{MAP_TITLES[key]}</div>"
                  + _map_cell(tiles[key], key, False, True)[len("<div class='eq-cell'>"):])
    return (f"<div class='eq-panel eq-card'><div class='eq-strip'>{cells}</div><div class='eq-note'>Dashed outline: the true defect. Number under each map: "
            "pixel AUC for finding it. ★ lowest-scoring region.</div></div>")


def comparison(results: list[dict]) -> str:
    """Verdict cards of two photographs and which of them has the lower poor-quality risk."""
    cards = ""
    for name, result in zip(("A", "B"), results):
        verdict = result["verdict"]
        cards += (f"<div class='eq-versus {verdict.lower()}'><span>Photograph {name}</span><b>{verdict}</b>"
                  f"<i>poor-quality risk {result['risk_bad']:.2f}</i></div>")
    a, b = (result["risk_bad"] for result in results)
    if abs(a - b) < 0.01:
        text = "The two photographs have practically the same predicted risk."
    else:
        text = f"Photograph {'A' if a < b else 'B'} has the lower predicted poor-quality risk ({min(a, b):.2f} against {max(a, b):.2f})."
    if {result["verdict"] for result in results} == {"PASS", "FAIL"}:
        text += " The decisions are opposite: a repeat acquisition changed the outcome."
    return f"<div class='eq-versus-row'>{cards}</div><div class='eq-headline center'>{text}</div>"


# ---------------------------------------------------------------- free-text request
def interpretation(criteria: Criteria) -> str:
    """Live preview of how the typed request is matched, before anything is computed."""
    if criteria.used_all and not criteria.matched and not criteria.excluded:
        chips = "<span class='eq-chip all'>All ten measurements</span>"
    elif criteria.used_all:
        chips = "<span class='eq-chip all'>All measurements" + (" except" if criteria.excluded else "") + "</span>"
    else:
        words = {name: ", ".join(criteria.matched.get(name, ())) for name in criteria.measurements}
        chips = "".join(f"<span class='eq-chip' title='matched: {esc(words[name])}'>{esc(DISPLAY_NAMES[name])}</span>"
                        for name in criteria.measurements)
    chips += "".join(f"<span class='eq-chip off' title='left out on request'>{esc(DISPLAY_NAMES[name])}</span>" for name in criteria.excluded)
    chips += "".join(f"<span class='eq-chip na' title='EyeQC measures technical quality only'>{esc(term)} · not assessed</span>"
                     for term in criteria.out_of_scope)
    return f"<div class='eq-chips'><em>Will report</em>{chips}</div>"


def reply(assessment: Assessment | None) -> str:
    """The answer to the request, assembled from the requested measurements only."""
    if assessment is None or assessment.reply is None:
        return ""
    found = assessment.reply
    request = assessment.request.strip()
    asked = f"<div class='eq-asked'>“{esc(request)}”</div>" if request else ""
    cards = ""
    for finding in found.findings:
        state = "low" if finding.low else "ok"
        area = "" if finding.low_area is None else (f"<span>{finding.map_key.capitalize()} map: {finding.low_area:.0%} of the retina "
                                                    "scores low</span>")
        advice = f"<span class='eq-advice'>{esc(finding.advice)}</span>" if finding.advice else ""
        cards += (f"<div class='eq-finding {state}'><div class='eq-finding-head'><b>{esc(finding.label)}</b><strong>{finding.value:.2f}</strong></div>"
                  f"<span class='eq-meter'><span style='width:{100 * finding.value:.0f}%'></span>"
                  f"<i style='left:{100 * min(finding.reference, 0.995):.1f}%' title='development reference {finding.reference:.2f}'></i></span>"
                  f"<span>{'below' if finding.low else 'at or above'} reference {finding.reference:.2f}</span>{area}{advice}</div>")
    missing = ""
    if found.not_assessed:
        missing = (f"<div class='eq-notassessed'>Not assessed: <b>{esc(', '.join(found.not_assessed))}</b>. EyeQC measures technical image "
                   "quality only; it does not examine anatomy or disease.</div>")
    verdict = assessment.result["verdict"].lower()
    return (f"<div class='eq-panel eq-reply'><div class='eq-reply-head'><div class='eq-title'>Response to your request</div>{asked}</div>"
            f"<div class='eq-headline'>{esc(found.headline)}</div>{missing}<div class='eq-findings'>{cards}</div>"
            f"<div class='eq-decision {verdict}'>{esc(found.decision)}</div></div>")
