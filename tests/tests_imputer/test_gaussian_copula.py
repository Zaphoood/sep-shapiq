"""Tests for the GaussianCopulaImputer class.

This module contains unit tests for the GaussianCopulaImputer class, including tests for
categorical feature detection, transformation methods, and imputation logic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from shapiq_student.imputer.gaussian.gaussian_copula_imputer import GaussianCopulaImputer

LOWER_BOUND = -4.0
UPPER_BOUND = 4.0
IMPUTED_LOWER = -2
IMPUTED_UPPER = 4


##############################################
# Tests for Transformation Methods --------- #
##############################################


def test_transform_to_gaussian(dummy_model) -> None:
    """Tests transforming background data to Gaussian space."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    expected_transformed_data = np.array(
        [
            [-0.674, -0.674, -0.674],
            [0, 0, 0],
            [0.674, 0.674, 0.674],
        ]
    )

    imputer = GaussianCopulaImputer(model=dummy_model, data=data)
    transformed_data = imputer._transform_to_gaussian(data)

    assert np.allclose(transformed_data, expected_transformed_data, atol=1e-3), (
        f"Expected {expected_transformed_data}, but got {transformed_data}"
    )


def test_transform_point_to_gaussian(dummy_model) -> None:
    """Test transforming a single point to Gaussian space."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x_test = np.array([8, 2, 1])
    # Features should be mapped to ranks [3, 1, 1], meaning quantiles [3/4, 1/4, 1/4]
    expected_quantiles = np.array([3 / 4, 1 / 4, 1 / 4])
    expected_x_transformed = norm.ppf(expected_quantiles)

    imputer = GaussianCopulaImputer(model=dummy_model, data=data)
    x_transformed = imputer._transform_point_to_gaussian(data, x_test)

    assert x_transformed.shape == x_test.shape
    assert np.allclose(x_transformed, expected_x_transformed, atol=1e-2)


def test_identity_transform(dummy_model) -> None:
    """Tests that transforming to Gaussian space and gives the original data."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [7.0, 2.5, 6.0],
            [2.0, 8.0, -1.0],
        ]
    )

    imputer = GaussianCopulaImputer(model=dummy_model, data=data)
    data_transformed = imputer._transform_to_gaussian(data)
    data_backtransformed = imputer._transform_from_gaussian(data_transformed)

    assert np.allclose(data, data_backtransformed)


def test_transform_to_original(dummy_model) -> None:
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

    original = imputer._transform_from_gaussian(gaussian_samples)

    # Check shape
    assert original.shape == gaussian_samples.shape

    # Check values are within original data range
    assert np.all(original >= np.min(data))
    assert np.all(original <= np.max(data))


##############################################
# Tests for Imputation --------------------- #
##############################################


def test_copula_imputation_single_feature_known(dummy_model) -> None:
    """Test imputation with a single feature known and two unknown."""
    # Create correlated data
    rng = np.random.default_rng(seed=42)
    data = rng.normal(size=(1000, 3))
    # Introduce correlation between features
    data[:, 1] = 0.8 * data[:, 0] + 0.2 * data[:, 1]
    data[:, 2] = 0.5 * data[:, 0] + 0.5 * data[:, 2]

    x_explain = np.array([1.0, np.nan, np.nan])
    coalitions = np.array([[True, False, False]])

    imputer = GaussianCopulaImputer(
        model=dummy_model, data=data, x=x_explain, n_mc_samples=1000, random_state=42
    )

    imputed = imputer._impute(x_explain, coalitions)

    # We expect a shape of (n_coaltions, n_features)
    assert imputed.shape == (coalitions.shape[0], x_explain.shape[0])

    # We can't predict exact values, but they should be within a reasonable range
    assert np.all(imputed[0, 1:] >= IMPUTED_LOWER)
    assert np.all(imputed[0, 1:] <= IMPUTED_UPPER)


def test_copula_imputer_value_function(dummy_model) -> None:
    """Test the value function of the copula imputer."""
    # Create correlated data
    rng = np.random.default_rng(seed=42)
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
    assert np.isscalar(result_value_function) or result_value_function.shape == (1,)
    assert IMPUTED_LOWER < result_value_function < IMPUTED_UPPER  # Reasonable range for imputed
