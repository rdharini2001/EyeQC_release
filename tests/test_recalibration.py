import numpy as np
import pytest

from eyeqc.contract import FAIL_THRESHOLD, PASS_THRESHOLD
from eyeqc.recalibration import Recalibration, fit_recalibration, labelled_participants, operating_metrics


def test_identity_mapping():
    risk = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(Recalibration(0.0, 1.0)(risk), risk)


def test_corrects_underestimated_risk():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.02, 0.98, 4000)
    labels = (rng.uniform(size=true.size) < true).astype(int)
    reported = true**2.5                       # systematically too low
    mapping = fit_recalibration(reported[:200], labels[:200])
    before = operating_metrics(labels[200:], reported[200:])
    after = operating_metrics(labels[200:], mapping(reported[200:]))
    assert mapping.slope >= 0
    assert after["ECE"] < before["ECE"] and after["Brier"] < before["Brier"]
    assert after["false_pass_rate"] < before["false_pass_rate"]


def test_monotone():
    rng = np.random.default_rng(1)
    risk = rng.uniform(size=300)
    mapping = fit_recalibration(risk, rng.integers(0, 2, 300))
    assert np.all(np.diff(mapping(np.linspace(0.01, 0.99, 50))) >= 0)


def test_needs_both_classes():
    with pytest.raises(ValueError):
        fit_recalibration([0.2, 0.3], [0, 0])


def test_thresholds_are_the_published_ones():
    assert round(PASS_THRESHOLD, 4) == 0.3453 and round(FAIL_THRESHOLD, 4) == 0.3861


def test_labelled_set_is_made_of_whole_participants():
    groups = np.repeat([f"p{i}" for i in range(40)], 4)
    chosen = labelled_participants(groups, 50, np.random.default_rng(0))
    assert chosen.sum() == 48
    assert not set(groups[chosen]) & set(groups[~chosen])
