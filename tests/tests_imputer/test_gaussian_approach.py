"""Tests for the GaussianImputer class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_student.imputer.exceptions import CategoricalFeatureError
from shapiq_student.imputer.gaussian import GaussianImputer


def test_check_categorical_features_valid():
    """Should pass silently when all features are continuous with >2 uniques."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["f1", "f2", "f3"],
        },
        "data": {
            "x_train": np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                ]
            ),
            "x_explain": np.array([[2.0, 3.0, 4.0]]),
        },
    }
    GaussianImputer(internal)  # No exception expected


def test_check_categorical_features_binary_integer():
    """Should raise ValueError naming column f2 when it has only 0/1 values."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["f1", "f2", "f3"],
        },
        "data": {
            "x_train": np.array(
                [
                    [1.0, 0, 3.0],
                    [2.0, 1, 4.0],
                    [3.0, 0, 5.0],
                ]
            ),
            "x_explain": np.array([[2.0, 1.0, 4.0]]),
        },
    }
    with pytest.raises(CategoricalFeatureError, match="f2") as exc:
        GaussianImputer(internal)
    msg = str(exc.value)
    assert "f2" in msg
    assert "f1" not in msg
    assert "f3" not in msg


def test_check_categorical_features_string():
    """Should raise ValueError naming column f2 when it contains strings."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["f1", "f2", "f3"],
        },
        "data": {
            "x_train": np.array(
                [
                    [1.0, "a", 3.0],
                    [2.0, "b", 4.0],
                    [3.0, "a", 5.0],
                ],
                dtype=object,
            ),
            "x_explain": np.array([[2.0, "b", 4.0]], dtype=object),
        },
    }
    with pytest.raises(CategoricalFeatureError, match="f2") as exc:
        GaussianImputer(internal)
    msg = str(exc.value)
    assert "f2" in msg
    assert "f1" not in msg
    assert "f3" not in msg


def test_check_categorical_features_mixed():
    """Should raise ValueError naming columns f2 and f3 for binary+string mix."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["f1", "f2", "f3", "f4"],
        },
        "data": {
            "x_train": np.array(
                [
                    [1.0, 0, "a", 3.0],
                    [2.0, 1, "b", 4.0],
                    [3.0, 0, "a", 5.0],
                ],
                dtype=object,
            ),
            "x_explain": np.array([[2.0, 1.0, "b", 4.0]], dtype=object),
        },
    }
    with pytest.raises(CategoricalFeatureError, match="f2.*f3") as exc:
        GaussianImputer(internal)
    msg = str(exc.value)
    # Both f2 (binary) and f3 (string) must be mentioned
    assert "f2" in msg
    assert "f3" in msg
    # f1 and f4 must not appear
    assert "f1" not in msg
    assert "f4" not in msg
