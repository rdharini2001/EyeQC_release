"""Quality phenotypes and cohort quality profiles.

The phenotype of a photograph that is not assigned PASS is the predefined measurement with the largest *risk-raising*
contribution to its score (positive term of the log-odds decomposition).  It describes the ten measurements only, not
the contribution of the visual features, and is not an adjudicated reason for rejection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import FEATURES

NONE = "none"


def quality_phenotype(contributions: dict) -> str:
    """Name of the measurement that raises poor-quality risk the most, or ``"none"`` if none raises it."""
    values = np.array([contributions[name] for name in FEATURES], float)
    return FEATURES[int(values.argmax())] if values.max() > 0 else NONE


def feature_contributions(features: pd.DataFrame, model) -> pd.DataFrame:
    """Calibrated log-odds contribution of each measurement for a table with one column per measurement."""
    values = features[list(FEATURES)].to_numpy(float)
    count = len(FEATURES)
    terms = (values - model.mean[:count]) / model.scale[:count] * model.coef[:count] * model.contract["platt_coefficient"]
    return pd.DataFrame(terms, columns=list(FEATURES), index=features.index)


def phenotype_composition(results: pd.DataFrame, cohort_column: str = "cohort") -> pd.DataFrame:
    """Share of each phenotype among photographs not assigned PASS, per cohort.

    ``results`` needs the columns ``verdict``, ``phenotype`` and ``cohort_column``.
    """
    flagged = results[results["verdict"] != "PASS"]
    return pd.crosstab(flagged[cohort_column], flagged["phenotype"], normalize="index")


def cohort_profiles(results: pd.DataFrame, cohort_column: str = "cohort") -> pd.DataFrame:
    """Median of each measurement per cohort."""
    return results.groupby(cohort_column)[list(FEATURES)].median()
