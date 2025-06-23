"""Tests for the GaussianApproach class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_student.imputer.gaussian_approach import CategoricalFeatureError, GaussianApproach


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
    GaussianApproach(internal)  # No exception expected


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
        GaussianApproach(internal)
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
        GaussianApproach(internal)
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
        GaussianApproach(internal)
    msg = str(exc.value)
    # Both f2 (binary) and f3 (string) must be mentioned
    assert "f2" in msg
    assert "f3" in msg
    # f1 and f4 must not appear
    assert "f1" not in msg
    assert "f4" not in msg


# Tests for Cov Mat and Mean Calculation ---


def test_calculate_mean_per_feature_valid():
    """Test mean calculation with valid data."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["f1", "f2", "f3"],
            "n_features": 3,
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
    approach = GaussianApproach(internal)
    expected_mean = np.mean(internal["data"]["x_train"], axis=0)
    approach.calculate_mean_per_feature()
    calculated_mean = approach.internal["parameters"]["mean_per_feature"]
    assert "mean_per_feature" in approach.internal["parameters"]
    np.testing.assert_array_almost_equal(calculated_mean, expected_mean)
    # Check if mean has correct shape
    assert calculated_mean.shape == (3,)


def test_calculate_mean_per_feature_empty_data():
    """Test mean calculation with empty data."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": [],
            "n_features": 0,
        },
        "data": {
            "x_train": np.array([]),
            "x_explain": np.array([]),
        },
    }
    with pytest.raises(ValueError, match="Training data is empty"):
        GaussianApproach(internal)


def test_calculate_mean_per_feature_invalid_data():
    """Test mean calculation with invalid data type."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["X1"],
            "n_features": 1,
        },
        "data": {
            "x_train": [1, 2, 3],  # Not a numpy array
            "x_explain": np.array([[1.0]]),
        },
    }
    with pytest.raises(AttributeError, match="object has no attribute 'T'"):
        GaussianApproach(internal)


def test_calculate_covariance_matrix_valid():
    """Test covariance matrix calculation with valid data."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["f1", "f2", "f3"],
            "n_features": 3,
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
    approach = GaussianApproach(internal)
    expected_cov = np.cov(internal["data"]["x_train"].T)
    approach.calculate_covariance_matrix()
    calculated_cov = approach.internal["parameters"]["cov_mat"]
    assert "cov_mat" in approach.internal["parameters"]
    np.testing.assert_array_almost_equal(calculated_cov, expected_cov)
    assert calculated_cov.shape == (3, 3)


def test_calculate_covariance_matrix_empty_data():
    """Test covariance calculation with empty data."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": [],
            "n_features": 0,
        },
        "data": {
            "x_train": np.array([]),
            "x_explain": np.array([]),
        },
    }
    with pytest.raises(ValueError, match="Training data is empty"):
        GaussianApproach(internal)


def test_calculate_covariance_matrix_invalid_data():
    """Test covariance calculation with invalid data type."""
    internal = {
        "parameters": {
            "verbose": {"progress": True},
            "approach": "gaussian",
            "feature_names": ["X1"],
            "n_features": 1,
        },
        "data": {
            "x_train": [1, 2, 3],  # Not a numpy array
            "x_explain": np.array([[1.0]]),
        },
    }
    with pytest.raises(AttributeError, match="object has no attribute 'T'"):
        GaussianApproach(internal)
