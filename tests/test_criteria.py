"""Free-form criteria are matched deterministically and never change the model."""
import pytest

from eyeqc.core import FEATURES
from eyeqc.criteria import DEFAULT_TEXT, parse_criteria, predefined_criteria
from eyeqc.maps import MAP_KEYS


def test_default_text_selects_the_named_measurements():
    found = parse_criteria(DEFAULT_TEXT)
    assert found.measurements == ("focus", "illumination_uniformity", "contrast", "artifact_burden")
    assert set(found.maps) == {"exposure", "contrast", "detail", "quality"} and not found.used_all


def test_negation_excludes():
    found = parse_criteria("check blur, ignore colour")
    assert "focus" in found.measurements and "color_balance" not in found.measurements
    only_excluding = parse_criteria("everything except clipping")
    assert only_excluding.measurements == tuple(name for name in FEATURES if name != "clipping")


@pytest.mark.parametrize("text", ["any hemorrhages?", "is there diabetic retinopathy", "check the optic disc and macula"])
def test_disease_and_anatomy_requests_are_declared_out_of_scope(text):
    found = parse_criteria(text)
    assert found.out_of_scope and found.used_all
    assert "Not assessed" in found.summary()


@pytest.mark.parametrize("text", ["", None, "please look at this one"])
def test_unrecognised_text_reports_everything(text):
    found = parse_criteria(text)
    assert found.measurements == FEATURES and found.maps == MAP_KEYS and found.used_all


def test_words_inside_other_words_do_not_match():
    assert "exposure" not in parse_criteria("dimensions of the file").matched
    assert "field_coverage" not in parse_criteria("airfield").matched


@pytest.mark.parametrize("text,expected", [
    ("is the photograph not sharp?", {"focus"}),
    ("is the image not in focus", {"focus"}),
    ("focus on glare", {"artifact_burden"}),
    ("do not check colour, check contrast", {"contrast"}),
    ("ignore colour and clipping. is there a shadow?", {"illumination_uniformity", "exposure"}),
    ("ignore colour, check blur", {"focus", "detail_uniformity"}),
    ("is the whole field visible?", {"field_coverage", "local_readability"}),
])
def test_questions_and_instructions(text, expected):
    assert set(parse_criteria(text).measurements) == expected


@pytest.mark.parametrize("text", ["is this gradable?", "should I retake it", "is it good enough"])
def test_overall_questions_report_everything(text):
    assert parse_criteria(text).measurements == FEATURES


def test_predefined():
    assert predefined_criteria().measurements == FEATURES
