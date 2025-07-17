"""Tests for the GaussianImputer class.

This module contains unit tests for the GaussianImputer class, including tests for categorical feature detection, mean and covariance calculations, and imputation logic.
"""

from __future__ import annotations

import numpy as np

from shapiq_student.imputer.gaussian.gaussian_imputer import GaussianImputer

##############################################
# Tests for Cov Mat and Mean Calculation --- #
##############################################


def test_calculate_mean_per_feature_valid(dummy_model) -> None:
    """Test mean calculation with valid data."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x = np.array([2.0, 3.0, 4.0])

    imputer = GaussianImputer(model=dummy_model, data=data, x=x)
    expected_mean = np.mean(data, axis=0)

    np.testing.assert_allclose(imputer.mean_per_feature, expected_mean)
    assert imputer.mean_per_feature.shape == (3,)


def test_calculate_covariance_matrix_valid(dummy_model) -> None:
    """Test covariance matrix calculation with valid data."""
    data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x = np.array([2.0, 3.0, 4.0])
    imputer = GaussianImputer(model=dummy_model, data=data, x=x)
    expected_cov = np.cov(data.T)
    np.testing.assert_allclose(imputer.cov_mat, expected_cov, atol=1e-5)
    assert imputer.cov_mat.shape == (3, 3)


##############################################
# Test for Imputation ---                    #
##############################################


def test_gaussian_imputation_first_feature_known_mean_and_cov_check() -> None:
    """Test imputation: first feature known, last two unknown; check mean and covariance."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    rng = np.random.default_rng(seed=42)
    x_train = rng.multivariate_normal(mean, cov, size=10000)

    # Check sample mean and covariance
    sample_mean = np.mean(x_train, axis=0)
    sample_cov = np.cov(x_train, rowvar=False)
    np.testing.assert_allclose(sample_mean, mean, atol=0.05)
    np.testing.assert_allclose(sample_cov, cov, atol=0.05)


def test_gaussian_imputation_first_feature_known(dummy_model) -> None:
    """Test imputation: first feature known, last two set to 1; check imputed mean."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    rng = np.random.default_rng(seed=42)
    x_train = rng.multivariate_normal(mean, cov, size=10000)
    x_explain = np.array([[1.0, np.nan, np.nan]])
    coalition = np.array([True, False, False])

    imputer = GaussianImputer(
        model=dummy_model,
        data=x_train,
        x=x_explain[0],
    )
    imputation_result = imputer._impute(x_explain[0], np.atleast_2d(coalition))
    imputed_features = imputation_result[0, ~coalition]

    # The mean should be close to [0.8, 0.5]
    np.testing.assert_allclose(imputed_features, [0.8, 0.5], atol=0.1)


def test_gaussian_imputer_value_function():
    """Test the vlaue function of the gaussian imputer."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    rng = np.random.default_rng(seed=42)
    x_train = rng.multivariate_normal(mean, cov, size=10000)
    x_explain = np.array([[1.0, np.nan, np.nan]])
    coalition = np.array([True, False, False])
    model = np.sum

    imputer = GaussianImputer(data=x_train, x=x_explain[0], model=model)

    result_value_function = imputer.value_function(np.atleast_2d(coalition))
    expected_sum = 1.0 + 0.8 + 0.5
    np.testing.assert_allclose(result_value_function, expected_sum, atol=0.1)
