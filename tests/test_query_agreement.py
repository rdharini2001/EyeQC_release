"""Free-text answers, reader agreement and same-eye pairs."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from eyeqc.agreement import fleiss_kappa, icc3, krippendorff_ordinal_alpha, reader_report
from eyeqc.core import FEATURES
from eyeqc.criteria import parse_criteria
from eyeqc.pairs import same_eye_summary
from eyeqc.query import answer
from eyeqc.recalibration import local_pass_threshold
from eyeqc.reference import GRADABLE_P10


def _result(**low):
    features = {name: 1.0 for name in FEATURES}
    features.update(low)
    return {"features": features, "verdict": "REVIEW", "risk_bad": 0.37}


def test_answer_reports_only_requested_measurements_and_never_changes_the_decision():
    reply = answer(parse_criteria("check contrast and glare"), _result(contrast=GRADABLE_P10["contrast"] / 2))
    assert [finding.name for finding in reply.findings] == ["contrast", "artifact_burden"]
    assert reply.findings[0].low and reply.findings[0].advice and not reply.findings[1].advice
    assert "1 of 2 requested measurements is below" in reply.headline
    assert "REVIEW" in reply.decision and "0.37" in reply.decision


def test_answer_declares_disease_questions_unassessed():
    assert answer(parse_criteria("is there glaucoma?"), _result()).not_assessed == ("glaucoma",)


def test_perfect_and_chance_agreement():
    grades = np.random.default_rng(0).integers(0, 3, 200)
    same = np.stack([grades] * 3, axis=1)
    assert fleiss_kappa(same) == pytest.approx(1) and krippendorff_ordinal_alpha(same) == pytest.approx(1)
    assert icc3(same)[0] == pytest.approx(1)
    noise = np.random.default_rng(1).integers(0, 3, (3000, 3))
    assert abs(fleiss_kappa(noise)) < 0.05 and abs(krippendorff_ordinal_alpha(noise)) < 0.05


def test_reader_report_matches_sklearn_and_accepts_words():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, 3, 60)
    grades = pd.DataFrame({name: np.array(["FAIL", "REVIEW", "PASS"])[np.clip(truth + (rng.random(60) < 0.2), 0, 2)] for name in ("A", "B")})
    model = np.array(["FAIL", "REVIEW", "PASS"])[truth]
    report = reader_report(grades, model, risk=1 - truth / 2, bootstraps=100)
    codes = grades["A"].map({"FAIL": 0, "REVIEW": 1, "PASS": 2})
    assert report["per_grader"].kappa_with_EyeQC[0] == pytest.approx(cohen_kappa_score(codes, truth, weights="quadratic"))
    assert report["kappa_matrix"].shape == (3, 3) and report["per_grader"].spearman_risk_vs_category.min() > 0.7
    with pytest.raises(ValueError):
        reader_report(grades.assign(A="maybe"), model)


def test_same_eye_pairs_use_eyes_with_exactly_two_photographs():
    frame = pd.DataFrame({"eye_id": ["a", "a", "b", "b", "c", "d", "d", "d"], "risk": [0.1, 0.9, 0.2, 0.25, 0.5, 0.1, 0.1, 0.1],
                          "verdict": ["PASS", "FAIL", "PASS", "PASS", "FAIL", "PASS", "PASS", "PASS"],
                          "artifact": [0, 4, 2, 0, 0, 0, 0, 0]})
    summary = same_eye_summary(frame, higher_is_worse={"artifact": True})
    assert summary["pairs"] == 2 and summary["opposite_PASS_FAIL"] == 0.5 and summary["same_category"] == 0.5
    assert summary["artifact_pairs"] == 2 and summary["artifact_higher_risk_for_worse"] == 0.5


def test_local_pass_threshold_keeps_poor_photographs_out_of_pass():
    risk = np.linspace(0.01, 0.99, 100)
    poor = (risk > 0.05).astype(int)
    cut = local_pass_threshold(risk, poor, sensitivity=0.9)
    assert np.mean(risk[poor == 1] > cut) >= 0.9
