"""Unit tests for CopulaApproach in shapiq_student.imputer.copula_approach."""

from __future__ import annotations

import numpy as np

from shapiq_student.imputer.copula_approach import CopulaApproach


def test_copula_approach_basic():
    """Test that CopulaApproach runs and returns finite results for dummy data."""
    # Dummy data
    x_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    x_explain = np.array([[0.15, 0.25], [0.35, 0.45]])
    internal = {
        "data": {
            "x_train": x_train,
            "x_explain": x_explain,
        },
        "parameters": {
            "feature_names": ["f1", "f2"],
            "n_explain": x_explain.shape[0],
            "n_features": x_train.shape[1],
            "n_MC_samples": 5,
            "verbose": [],
            "approach": "copula",
        },
        "iter_list": [{}],
    }
    approach = CopulaApproach(internal)
    result = approach.copula_imputation()
    # The result should be a 3D array: (n_MC_samples, n_explain * n_coalitions, n_features)
    n_coalitions = 2 ** x_train.shape[1]
    assert result.shape == (5, 2 * n_coalitions, 2)
    # Check that the result is finite
    assert np.all(np.isfinite(result))
