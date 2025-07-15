"""Tests for the GaussianCopulaImputer class.

This module contains unit tests for the GaussianCopulaImputer class, including tests for
categorical feature detection, transformation methods, and imputation logic.
"""

# # TODO (milanagm): REVIEW WHOLE TEST SCRIPT

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shapiq_student.imputer.gaussian.copula_imputer import GaussianCopulaImputer
from shapiq_student.imputer.gaussian.exceptions import CategoricalFeatureError

MIN_REASONABLE_VALUE = -5
MAX_REASONABLE_VALUE = 5
DATA_MIN = 1
DATA_MAX = 9
LOWER_BOUND = 0.5
UPPER_BOUND = 3.0


def dummy_model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A simple placeholder model for testing.

    Args:
        x: Input data.

    Returns:
        Sum over the last axis of the input.
    """
    return np.asarray(np.sum(x, axis=-1), dtype=float)


def test_check_categorical_features_valid() -> None:
    """Test that no error is raised when all features are continuous with >2 unique values."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x = np.array([[2.0, 3.0, 4.0]])
    # Should not raise error
    GaussianCopulaImputer(model=dummy_model, data=data, x=x)


def test_check_categorical_features_binary_integer() -> None:
    """Test that CategoricalFeatureError is raised for binary integer columns (e.g., 0/1)."""
    data = np.array(
        [
            [1.0, 0, 3.0],
            [2.0, 1, 4.0],
            [3.0, 0, 5.0],
        ]
    )
    x = np.array([2.0, 1.0, 4.0])
    with pytest.raises(CategoricalFeatureError) as exc:
        GaussianCopulaImputer(model=dummy_model, data=data, x=x)
    msg = str(exc.value)
    assert "f2" in msg  # The second column (index 1) should be flagged as categorical
    assert "f1" not in msg
    assert "f3" not in msg


def test_check_categorical_features_string() -> None:
    """Test that CategoricalFeatureError is raised for columns containing strings."""
    data = np.array(
        [
            [1.0, "a", 3.0],
            [2.0, "b", 4.0],
            [3.0, "a", 5.0],
        ],
        dtype=object,
    )
    x = np.array([2.0, "b", 4.0], dtype=object)
    with pytest.raises(CategoricalFeatureError) as exc:
        GaussianCopulaImputer(model=dummy_model, data=data, x=x)
    msg = str(exc.value)
    assert "f2" in msg
    assert "f1" not in msg
    assert "f3" not in msg


##############################################
# Tests for Transformation Methods          #
##############################################


def test_rank_gaussian_transform() -> None:
    """Test the rank-Gaussian transformation preserves ranks and produces standard normal."""
    # Use more data points to ensure empirical std is close to 1
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 10, size=(100, 3))
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)
    transformed = imputer.rank_gaussian_transform(data)

    # Check shape
    assert transformed.shape == data.shape

    # Check each column is standard normal distributed
    for col in transformed.T:
        assert np.allclose(np.mean(col), 0, atol=1e-2)
        assert np.allclose(np.std(col), 1, atol=0.05)

    # Check ranks are preserved
    for i in range(data.shape[1]):
        original_ranks = np.argsort(data[:, i])
        transformed_ranks = np.argsort(transformed[:, i])
        assert np.array_equal(original_ranks, transformed_ranks)


def test_transform_point_to_gaussian() -> None:
    """Test transforming a single point to Gaussian space."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x_point = np.array([2.5, 3.5, 4.5])
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    transformed = imputer.transform_point_to_gaussian(x_point, data)

    # Check shape
    assert transformed.shape == x_point.shape

    assert np.all(transformed >= DATA_MIN)
    assert np.all(transformed <= DATA_MAX)


def test_transform_to_original() -> None:
    """Test transforming from Gaussian space back to original space."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    gaussian_samples = np.array([[0, 0, 0], [1, -1, 0.5], [-1, 1, -0.5]])
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    original = imputer.transform_to_original(gaussian_samples)

    # Check shape
    assert original.shape == gaussian_samples.shape

    # Check values are within original data range
    assert np.all(original >= DATA_MIN)
    assert np.all(original <= DATA_MAX)


##############################################
# Tests for Imputation                      #
##############################################


def test_copula_imputation_first_feature_known() -> None:
    """Test imputation with first feature known, others unknown."""
    # Create correlated data
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1000, 3))
    data[:, 1] = 0.8 * data[:, 0] + 0.2 * data[:, 1]  # Create correlation
    data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 2]  # Create correlation

    # Create explanation point (first feature at 1.0)
    x_explain = np.array([1.0, np.nan, np.nan])
    coalition = np.array([True, False, False])

    imputer = GaussianCopulaImputer(
        model=dummy_model, data=data, x=x_explain, n_mc_samples=1000, random_state=42
    )

    # Get imputed values
    imputation_result = imputer.impute(x_explain, np.atleast_2d(coalition))

    # The expected value should be sum of known feature (1.0) plus the conditional means
    # For the copula approach, we can't predict exact values but should be reasonable
    assert LOWER_BOUND < imputation_result[0] < UPPER_BOUND


def test_copula_imputer_value_function() -> None:
    """Test the value function of the copula imputer."""
    # Create correlated data
    rng = np.random.default_rng(42)
    data = rng.normal(size=(1000, 3))
    data[:, 1] = 0.8 * data[:, 0] + 0.2 * data[:, 1]  # Create correlation
    data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 2]  # Create correlation

    # Create explanation point (first feature at 1.0)
    x_explain = np.array([1.0, np.nan, np.nan])
    coalition = np.array([True, False, False])

    imputer = GaussianCopulaImputer(
        model=dummy_model, data=data, x=x_explain, n_mc_samples=1000, random_state=42
    )

    result_value_function = imputer.value_function(np.atleast_2d(coalition))

    # The expected value should be sum of known feature (1.0) plus the conditional means
    # For the copula approach, we can't predict exact values but should be reasonable
    assert LOWER_BOUND < result_value_function < UPPER_BOUND


def test_null_point_error() -> None:
    """Test that error is raised when x is None."""
    rng = np.random.default_rng()
    data = rng.random((10, 2))
    imputer = GaussianCopulaImputer(model=dummy_model, data=data)

    with pytest.raises(ValueError, match="Explanation point x cannot be None"):
        imputer.impute(None, np.array([[True, False]]))
