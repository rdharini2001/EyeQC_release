"""Everything the interface shows for one photograph, computed with the released EyeQC package (no UI code here)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import zipfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eyeqc.backbone import load_backbone
from eyeqc.core import ACTION_RULES, DISPLAY_NAMES, FEATURES, extract_features, scope_check
from eyeqc.criteria import Criteria
from eyeqc.experiments import apply_degradation
from eyeqc.maps import MAP_KEYS, lowest_region, overlay, pixel_auc, quality_maps
from eyeqc.model import load_model
from eyeqc.phenotypes import NONE, quality_phenotype
from eyeqc.pipeline import read_rgb
from eyeqc.query import Answer, answer

MAX_PIXELS = 40_000_000
DEMO_DEFECTS = (("glare", "Glare", 7), ("shadow", "Shadow", 11))    # (degradation, row label, fixed random seed)
MODEL = load_model()
CACHE_SIZE = 8
_CACHE: OrderedDict = OrderedDict()
_LOCK = threading.Lock()


@dataclass
class MapTile:
    image: np.ndarray                       # RGBA, square crop around the retina
    value: str                              # number shown under the map
    outline: list | None = None             # polygon(s) of a simulated defect, in [0, 1] crop coordinates
    star: tuple | None = None               # lowest-scoring region, in [0, 1] crop coordinates


@dataclass
class Assessment:
    name: str
    result: dict
    criteria: Criteria
    photo: np.ndarray
    rows: dict[str, dict[str, MapTile]] = field(default_factory=dict)
    in_scope: bool = True
    reply: Answer | None = None
    request: str = ""
    seconds: float = 0.0                    # measurements + DINOv2 + score, without the maps

    @property
    def features(self) -> dict:
        return self.result["features"]

    @property
    def phenotype(self) -> str:
        return quality_phenotype(self.result["axis_contributions"])

    def next_step(self) -> tuple[str, str]:
        """(headline, detail) - an illustrative suggestion derived from the verdict and the largest risk-raising measurement."""
        verdict = self.result["verdict"]
        if verdict == "PASS":
            return "Use", "Suitable at the current threshold."
        headline = "Reacquire" if verdict == "FAIL" else "Review"
        if self.phenotype == NONE:
            return headline, "No single measurement raises the risk; inspect the photograph and its maps."
        return headline, f"{DISPLAY_NAMES[self.phenotype]} raises the risk most. {ACTION_RULES[self.phenotype]}"


def warm_up() -> None:
    """Load the DINOv2 encoder before the first request, so reported processing times do not include loading it."""
    try:
        load_backbone()
    except Exception as problem:  # no network yet: the first request tries again and reports the error
        print(f"The encoder could not be loaded at start-up: {problem}", flush=True)


def read_photo(path: str | Path) -> np.ndarray:
    with Image.open(path) as handle:
        if handle.width * handle.height > MAX_PIXELS:
            raise ValueError("The photograph is larger than 40 megapixels.")
    return read_rgb(path)


def square_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.51) + 1
    return cy - half, cy + half, cx - half, cx + half


def crop(array: np.ndarray, box: tuple, fill=0) -> np.ndarray:
    y0, y1, x0, x1 = box
    height, width = array.shape[:2]
    out = np.full((y1 - y0, x1 - x0) + array.shape[2:], fill, dtype=array.dtype)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, height), max(x0, 0), min(x1, width)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def tile(features: dict, score: np.ndarray, style: str, box: tuple, value: str, defect: np.ndarray | None, size: int = 240) -> MapTile:
    image = cv2.resize(crop(overlay(features, score, style=style), box), (size, size), interpolation=cv2.INTER_AREA)
    side = box[1] - box[0]
    star = lowest_region(score)
    if star is not None:
        star = ((star[1] - box[2]) / side, (star[0] - box[0]) / side)
    outline = None
    if defect is not None:
        contours, _ = cv2.findContours(crop(defect.astype(np.uint8), box), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = [(c[:, 0, :] / side).tolist() for c in contours if len(c) > 2]
    return MapTile(image=image, value=value, outline=outline, star=star)


def map_rows(rgb: np.ndarray, features: dict, style: str = "spectral", demo: bool = True,
             maps: dict | None = None) -> dict[str, dict[str, MapTile]]:
    """Maps of the photograph, plus (optionally) maps of the same photograph with simulated glare and shadow."""
    rows = {}
    maps = maps or quality_maps(features)
    box = square_box(features["support_mask"])
    rows["This photo"] = {}
    for key in MAP_KEYS:
        score = maps[key][features["inner_mask"]]
        low = float((score < 0.5).mean()) if score.size else float("nan")
        rows["This photo"][key] = tile(features, maps[key], style, box, f"{low:.0%} low", None)
    if not demo:
        return rows
    for kind, label, seed in DEMO_DEFECTS:
        degraded, mask = apply_degradation(rgb, kind, 1.0, np.random.default_rng(seed))
        changed = extract_features(degraded)
        changed_maps = quality_maps(changed)
        truth = cv2.resize(mask.astype(np.uint8), changed["inner_mask"].shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        changed_box = square_box(changed["support_mask"])
        rows[label] = {}
        for key in MAP_KEYS:
            auc = pixel_auc(changed_maps[key], truth)
            rows[label][key] = tile(changed, changed_maps[key], style, changed_box, "–" if np.isnan(auc) else f"{auc:.2f}", truth)
    return rows


def _analyse(path: str | Path, style: str, demo: bool) -> dict:
    """Model result and map images of one photograph, cached so that changing only the request is immediate."""
    stat = os.stat(path)
    key = (str(path), stat.st_mtime_ns, stat.st_size, style, demo)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    rgb = read_photo(path)
    result = MODEL.analyze(rgb)
    features = result["features"]
    maps = quality_maps(features)
    entry = {"result": result, "maps": maps, "photo": crop(features["analysis_rgb"], square_box(features["support_mask"])),
             "rows": map_rows(rgb, features, style, demo, maps), "in_scope": bool(scope_check(features)["in_scope"])}
    with _LOCK:
        _CACHE[key] = entry
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return entry


def assess(path: str | Path, criteria: Criteria, style: str = "spectral", demo: bool = True, request: str = "") -> Assessment:
    entry = _analyse(path, style, demo)
    result = entry["result"]
    return Assessment(name=Path(path).name, result=result, criteria=criteria, photo=entry["photo"], rows=entry["rows"],
                      in_scope=entry["in_scope"], reply=answer(criteria, result, entry["maps"]), request=request,
                      seconds=float(result["runtime_end_to_end_sec"]))


def export_report(assessment: Assessment) -> str:
    """Zip with the numbers (JSON) and the maps (PNG) of one assessment."""
    result = assessment.result
    report = {
        "file": assessment.name,
        "model_version": result["model_version"],
        "poor_quality_risk": result["risk_bad"],
        "verdict": result["verdict"],
        "measurements": {name: float(assessment.features[name]) for name in FEATURES},
        "log_odds": {"intercept": result["base_calibrated_logit"], **result["axis_contributions"],
                     "visual_features": result["representation_contribution"], "total": result["calibrated_logit"]},
        "criteria": {"request": assessment.request, "reported": list(assessment.criteria.measurements), "summary": assessment.criteria.summary()},
        "answer": None if assessment.reply is None else asdict(assessment.reply),
        "processing_seconds": round(assessment.seconds, 3),
        "suggestion": " - ".join(assessment.next_step()),
        "note": "Research use only. Suggestions are illustrative and not validated clinical recommendations.",
    }
    target = Path(tempfile.mkdtemp(prefix="eyeqc_")) / f"eyeqc_report_{Path(assessment.name).stem}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("report.json", json.dumps(report, indent=2))
        for row, tiles in assessment.rows.items():
            for key, tile in tiles.items():
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(tile.image, cv2.COLOR_RGBA2BGRA))
                if ok:
                    bundle.writestr(f"maps/{row.lower().replace(' ', '_')}_{key}.png", encoded.tobytes())
    return str(target)
