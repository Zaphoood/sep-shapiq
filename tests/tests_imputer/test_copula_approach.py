"""Tests for the GaussianCopulaImputer class.

This module contains unit tests for the GaussianCopulaImputer class, including tests for categorical feature detection,
Gaussian transformation, and imputation logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shapiq_student.imputer.gaussian.copula_imputer import GaussianCopulaImputer
from shapiq_student.imputer.gaussian.exceptions import CategoricalFeatureError

MIN_EXPECTED_SD = 0.7
MAX_EXPECTED_SD = 1.3
SHAPIRO_SAMPLE_LIMIT = 5000
SHAPIRO_SIGNIFICANCE = 0.05
NORMAL_BOUND = 5


def dummy_model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A simple placeholder model for testing.

    Args:
        x (np.ndarray[Any, Any]): Input data.

    Returns:
        np.ndarray[Any, Any]: Sum over the last axis of the input.
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
# Tests for Gaussian Transformation ---     #
##############################################


def test_gaussian_transform_valid() -> None:
    """Test that Gaussian transformation produces standard normal distributions."""
    # Use larger dataset for more stable statistics
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
        ]
    )
    x = np.array([2.0, 3.0, 4.0])
    imputer = GaussianCopulaImputer(model=dummy_model, data=data, x=x)

    transformed = imputer.gaussian_transform(data)

    # Check shape preserved
    assert transformed.shape == data.shape

    # Check each column is approximately standard normal
    for col_idx, col in enumerate(transformed.T):
        # Mean should be close to 0 (rank transformation centers data)
        assert pytest.approx(np.mean(col), abs=0.2) == 0

        # Standard deviation check with named constants
        col_sd = np.std(col)
        assert MIN_EXPECTED_SD <= col_sd <= MAX_EXPECTED_SD, f"Column {col_idx} has SD {col_sd}"

        # Test normality using Shapiro-Wilk
        if len(col) <= SHAPIRO_SAMPLE_LIMIT:
            from scipy.stats import shapiro

            _, p_value = shapiro(col)
            assert p_value > SHAPIRO_SIGNIFICANCE, (
                f"Column {col_idx} fails normality test (p={p_value})"
            )


def test_transform_x_explain() -> None:
    """Test transformation of explanation point using training data's ECDF."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x = np.array([2.0, 3.0, 4.0])
    imputer = GaussianCopulaImputer(model=dummy_model, data=data, x=x)

    transformed = imputer.transform_x_explain(x, data)

    assert transformed.shape == x.shape
    # Check values are within reasonable bounds for standard normal
    assert np.all(np.isfinite(transformed))
    assert np.all(np.abs(transformed) < NORMAL_BOUND)


def test_inverse_transform() -> None:
    """Test that inverse transformation returns to original data range."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x = np.array([2.0, 3.0, 4.0])
    imputer = GaussianCopulaImputer(model=dummy_model, data=data, x=x)

    # Create some standard normal samples using new Generator interface
    rng = np.random.default_rng()
    z_samples = rng.normal(0, 1, size=(10, data.shape[1]))
    original = imputer.inverse_transform(z_samples)

    assert original.shape == z_samples.shape
    # Values should be in original data range
    assert np.all(original >= np.min(data, axis=0))
    assert np.all(original <= np.max(data, axis=0))


##############################################
# Test for Imputation ---                    #
##############################################


def test_copula_imputation_first_feature_known() -> None:
    """Test imputation: first feature known, last two unknown; check imputed values."""
    # Create correlated data
    rng = np.random.default_rng(42)
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    x_train = rng.multivariate_normal([0, 0, 0], cov, size=10000)

    # Explanation point with first feature at 1 (in original space)
    x_explain = np.array([1.0, np.nan, np.nan])
    coalition = np.array([True, False, False])

    imputer = GaussianCopulaImputer(
        model=dummy_model, data=x_train, x=x_explain, n_mc_samples=1000, random_state=42
    )

    # Get imputed values (mean predictions)
    imputation_result = imputer.impute(x_explain, np.atleast_2d(coalition))

    # Transform explanation point to check expected values in Gaussian space
    x_transformed = imputer.transform_x_explain(x_explain, x_train)
    expected_cond_mean = x_transformed[0] * np.array([0.8, 0.5])  # cov[0,1:] * x_value

    # Transform back to original space
    expected_values = imputer.inverse_transform(
        np.atleast_2d(np.concatenate([[x_transformed[0]], expected_cond_mean]))
    )[0, 1:]

    # Check imputed values are close to expected
    np.testing.assert_allclose(
        imputation_result[0], np.sum(np.concatenate([[x_explain[0]], expected_values])), atol=0.1
    )


def test_copula_imputer_value_function() -> None:
    """Test the value function of the copula imputer."""
    rng = np.random.default_rng(42)
    cov = np.array([[1, 0.8], [0.8, 1]])
    x_train = rng.multivariate_normal([0, 0], cov, size=10000)
    x_explain = np.array([1.0, np.nan])
    coalition = np.array([True, False])

    imputer = GaussianCopulaImputer(
        data=x_train, x=x_explain, model=dummy_model, n_mc_samples=1000, random_state=42
    )

    result_value_function = imputer.value_function(np.atleast_2d(coalition))

    # Expected value is x_explain[0] + E[x2|x1=transformed(1.0)]
    x_transformed = imputer.transform_x_explain(x_explain, x_train)
    expected_cond_mean = x_transformed[0] * 0.8  # cov[0,1] * x_value
    expected_x2 = imputer.inverse_transform(np.array([[0, expected_cond_mean]]))[0, 1]
    expected_sum = x_explain[0] + expected_x2

    np.testing.assert_allclose(result_value_function, expected_sum, atol=0.1)
