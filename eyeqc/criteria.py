"""Free-form quality criteria.

A typed request such as ``"check focus, glare and uneven illumination"`` is matched against a fixed vocabulary and
turned into a selection of EyeQC measurements and maps to report.  Matching is deterministic keyword matching, not a
language model.  It changes **only what is displayed**: the risk score and the PASS / REVIEW / FAIL decision always use
all ten measurements and the visual features.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .core import DISPLAY_NAMES, FEATURES
from .maps import MAP_KEYS

# measurement -> (regular expressions that refer to it, maps that help inspect it)
VOCABULARY: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "focus": ((r"focus\w*", r"de-?focus\w*", r"sharp\w*", r"blur\w*"), ("detail",)),
    "detail_uniformity": ((r"detail\w*", r"textur\w*", r"motion", r"smear\w*", r"blur\w*"), ("detail",)),
    "local_readability": ((r"readab\w*", r"legib\w*", r"visib\w*", r"vessel\w*"), ("quality", "detail")),
    "illumination_uniformity": ((r"illuminat\w*", r"light\w*", r"uneven\w*", r"shadow\w*", r"vignett\w*"), ("exposure",)),
    "exposure": ((r"exposure", r"(?:over|under)[- ]?expos\w*", r"dark\w*", r"bright\w*", r"shadow\w*", r"dim"), ("exposure",)),
    "contrast": ((r"contrast", r"haz[ey]\w*", r"wash(?:ed)?[- ]?out", r"veil\w*", r"media opacit\w*", r"cataract\w*"), ("contrast",)),
    "field_coverage": ((r"field", r"coverage", r"cent(?:er|re)(?:d|ing)?", r"de-?cent(?:er|r)\w*", r"crop\w*", r"framing"), ()),
    "artifact_burden": ((r"art[ie]fact\w*", r"glare", r"reflect\w*", r"flare", r"dust", r"(?:eye)?lash\w*", r"spots?"), ("exposure", "detail")),
    "clipping": ((r"clip\w*", r"saturat\w*", r"highlights?", r"crushed"), ("exposure",)),
    "color_balance": ((r"colou?r\w*", r"white balance", r"tint\w*", r"hue"), ()),
}
GENERAL_TERMS = (r"overall", r"(?:image )?quality", r"everything", r"all (?:criteria|measurements|concepts)", r"gradab\w*", r"usab\w*",
                 r"acceptab\w*", r"good enough", r"retake", r"re-?acquire")
# Requests EyeQC cannot answer: it assesses technical image quality, not anatomy or disease.
OUT_OF_SCOPE = (r"diabet\w*", r"retinopathy", r"\bdr\b", r"h(?:a)?emorrhage\w*", r"exudate\w*", r"micro-?aneurysm\w*", r"glaucoma\w*",
                r"drusen", r"amd", r"macular degeneration", r"o?edema", r"disease", r"patholog\w*", r"lesion\w*", r"tumou?r\w*",
                r"cup[- ]to[- ]disc", r"optic (?:disc|nerve)", r"macula\w*", r"fovea\w*")
# Only explicit instructions exclude a measurement: "is the photograph not sharp?" must still report focus.
NEGATIONS = (r"\b(?:ignor(?:e|ing)|except|exclud(?:e|ing)|skip(?:ping)?|leave out|leaving out|not including|without (?:checking|assessing|reporting)|"
             r"(?:do not|don'?t|no need to) (?:check|report|assess|include))\b")
# A new clause, which ends the reach of an exclusion.
CLAUSE_BREAKS = r"[.;?!\n]|\bbut\b|(?=\b(?:check|assess|report|evaluate|measure|look at|include)\b(?! (?:or|and)\b))"
DEFAULT_TEXT = "Assess focus, illumination, contrast and artifacts."


@dataclass(frozen=True)
class Criteria:
    """What to report for one assessment."""

    measurements: tuple[str, ...]
    maps: tuple[str, ...]
    matched: dict[str, tuple[str, ...]] = field(default_factory=dict)
    excluded: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    used_all: bool = False

    def summary(self) -> str:
        """One or two plain sentences describing how the request was interpreted."""
        if self.used_all and not self.matched:
            lead = "All ten predefined measurements are reported."
        else:
            lead = "Reporting: " + ", ".join(DISPLAY_NAMES[name] for name in self.measurements) + "."
        notes = [lead]
        if self.excluded:
            notes.append("Left out on request: " + ", ".join(DISPLAY_NAMES[name] for name in self.excluded) + ".")
        if self.out_of_scope:
            notes.append("Not assessed (EyeQC measures technical quality only): " + ", ".join(self.out_of_scope) + ".")
        return " ".join(notes)


def predefined_criteria() -> Criteria:
    """All ten measurements and all maps."""
    return Criteria(measurements=FEATURES, maps=MAP_KEYS, used_all=True)


def _find(patterns: tuple[str, ...], text: str) -> list[re.Match]:
    return [match for pattern in patterns for match in re.finditer(rf"(?<![\w-])(?:{pattern})(?![\w-])", text)]


def parse_criteria(text: str | None) -> Criteria:
    """Interpret a free-form request.

    A term that follows "ignore", "except", "exclude", "skip", "leave out" or "do not check" in the same clause excludes
    its measurement.
    If nothing is recognised, all ten measurements are reported (and ``used_all`` is true).
    """
    lowered = (text or "").lower()
    out_of_scope = tuple(sorted({match.group(0) for match in _find(OUT_OF_SCOPE, lowered)}))
    included: dict[str, set[str]] = {}
    excluded: set[str] = set()
    lowered_clauses = re.sub(r"\bfocus(?:ing)? on\b", "check", lowered)          # "focus on glare" is not a request about focus
    lowered_clauses = re.sub(r"\b(do not|don'?t|no need to) (check|report|assess|include)\b", "skip", lowered_clauses)
    for clause in re.split(CLAUSE_BREAKS, lowered_clauses):
        negation = re.search(NEGATIONS, clause)
        for name, (patterns, _) in VOCABULARY.items():
            for match in _find(patterns, clause):
                if negation and match.start() > negation.start():
                    excluded.add(name)
                else:
                    included.setdefault(name, set()).add(match.group(0))
    excluded -= set(included)
    wants_all = bool(_find(GENERAL_TERMS, lowered_clauses))
    if included and not wants_all:
        measurements = tuple(name for name in FEATURES if name in included)
    else:
        measurements = tuple(name for name in FEATURES if name not in excluded)
    if len(measurements) == len(FEATURES):
        maps = MAP_KEYS
    else:
        wanted = {key for name in measurements for key in VOCABULARY[name][1]} | {"quality"}    # the combined map is the overview
        maps = tuple(key for key in MAP_KEYS if key in wanted)
    return Criteria(measurements=measurements, maps=maps, matched={k: tuple(sorted(v)) for k, v in included.items()},
                    excluded=tuple(name for name in FEATURES if name in excluded), out_of_scope=out_of_scope,
                    used_all=wants_all or not included)
