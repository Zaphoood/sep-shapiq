"""Tests for the GaussianImputer class.

This module contains unit tests for the GaussianImputer class, including tests for categorical feature detection, mean and covariance calculations, and imputation logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shapiq_student.imputer.gaussian.exceptions import CategoricalFeatureError
from shapiq_student.imputer.gaussian.gaussian_imputer import GaussianImputer


def dummy_model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A simple placeholder model for testing.

    Args:
        x (np.ndarray[Any, Any]): Input data.

    Returns:
        np.ndarray[Any, Any]: Sum over the last axis of the input.
    """
    return np.asarray(np.sum(x, axis=-1), dtype=float)


class TestInputValidation:
    """Tests that input data is handled correctly and that malformed data is detected successfully."""

    def test_categorical_feature_check_only_continuous(self) -> None:
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
        GaussianImputer(model=dummy_model, data=data, x=x)

    def test_categorical_feature_check_binary_column(self) -> None:
        """Test that an exception is raised if the background data has column containing only two unique values."""
        data = np.array(
            [
                [1.0, 0, 3.0],
                [2.0, 1, 4.0],
                [3.0, 0, 5.0],
            ]
        )
        x = np.array([2.0, 1.0, 4.0])
        with pytest.raises(CategoricalFeatureError) as exc:
            GaussianImputer(model=dummy_model, data=data, x=x)
        msg = str(exc.value)
        # The second column (index 1) should be flagged as categorical, so 'f2' should be in the message
        assert "f2" in msg
        # The first and third columns should not be flagged, so 'f1' and 'f3' should not be in the message
        assert "f1" not in msg
        assert "f3" not in msg

    def test_categorical_feature_check_string(self) -> None:
        """Test that an exception is raised if the background data has a column containing string values."""
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
            GaussianImputer(model=dummy_model, data=data, x=x)
        msg = str(exc.value)
        assert "f2" in msg
        assert "f1" not in msg
        assert "f3" not in msg

    def test_categorical_feature_check_mixed(self) -> None:
        """Test that an exception is raised if the background data has binary and string-valued columns."""
        data = np.array(
            [
                [1.0, 0, "a", 3.0],
                [2.0, 1, "b", 4.0],
                [3.0, 0, "a", 5.0],
            ],
            dtype=object,
        )
        x = np.array([2.0, 1.0, "b", 4.0], dtype=object)
        with pytest.raises(CategoricalFeatureError) as exc:
            GaussianImputer(model=dummy_model, data=data, x=x)
        msg = str(exc.value)
        # Both f2 (index 1) and f3 (index 2) must be mentioned
        assert "f2" in msg
        assert "f3" in msg
        # f1 and f4 must not appear
        assert "f1" not in msg
        assert "f4" not in msg

    def test_x_explain_shapes(self):
        """Tests that an explain point can be passed both as a vector and a matrix with one row; both when passing in the constructor and when calling the fit() method."""
        data = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        x_explain = np.array([3, 2, 1])
        coalitions = np.array([[False, True, False]])

        imputer = GaussianImputer(model=dummy_model, data=data, x=x_explain.copy())
        imputer.value_function(coalitions)

        imputer = GaussianImputer(model=dummy_model, data=data, x=np.atleast_2d(x_explain.copy()))
        imputer.value_function(coalitions)

        imputer = GaussianImputer(model=dummy_model, data=data)
        imputer.fit(x_explain.copy())
        imputer.value_function(coalitions)

        imputer = GaussianImputer(model=dummy_model, data=data)
        imputer.fit(np.atleast_2d(x_explain.copy()))
        imputer.value_function(coalitions)


##############################################
# Tests for Cov Mat and Mean Calculation --- #
##############################################


def test_calculate_mean_per_feature_valid() -> None:
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
    np.testing.assert_array_almost_equal(imputer.mean_per_feature, expected_mean)
    assert imputer.mean_per_feature.shape == (3,)


def test_calculate_mean_per_feature_empty_data() -> None:
    """Test that EmptyDataError is raised when calculating mean with empty data."""
    data = np.empty((0, 3))
    x = np.array([])
    with pytest.raises(ValueError, match="data.*empty"):
        GaussianImputer(model=dummy_model, data=data, x=x)


def test_calculate_covariance_matrix_valid() -> None:
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
    np.testing.assert_array_almost_equal(imputer.cov_mat, expected_cov)
    assert imputer.cov_mat.shape == (3, 3)


def test_calculate_covariance_matrix_empty_data() -> None:
    """Test that EmptyDataError is raised when calculating covariance with empty data."""
    data = np.empty((0, 3))
    x = np.array([])
    with pytest.raises(ValueError, match="data.*empty"):
        GaussianImputer(model=dummy_model, data=data, x=x)


##############################################
# Test for Imputation ---                    #
##############################################


def test_gaussian_imputation_first_feature_known_mean_and_cov_check() -> None:
    """Test imputation: first feature known, last two unknown; check mean and covariance."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    rng = np.random.default_rng()
    x_train = rng.multivariate_normal(mean, cov, size=10000)

    # Check sample mean and covariance
    sample_mean = np.mean(x_train, axis=0)
    sample_cov = np.cov(x_train, rowvar=False)
    np.testing.assert_allclose(sample_mean, mean, atol=0.05)
    np.testing.assert_allclose(sample_cov, cov, atol=0.05)


def test_gaussian_imputation_first_feature_known() -> None:
    """Test imputation: first feature known, last two set to 1; check imputed mean."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    # TODO(Zaphoood): Fix seed to make tests deterministic
    rng = np.random.default_rng()
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
    np.testing.assert_allclose(imputed_features, [0.8, 0.5], atol=0.05)


def test_gaussian_imputer_value_function():
    """Test the vlaue function of the gaussian imputer."""
    # TODO (milanagm): set seed o ä für die tests adden
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    rng = np.random.default_rng()
    x_train = rng.multivariate_normal(mean, cov, size=10000)
    x_explain = np.array([[1.0, np.nan, np.nan]])
    coalition = np.array([True, False, False])
    model = np.sum

    imputer = GaussianImputer(data=x_train, x=x_explain[0], model=model)

    result_value_function = imputer.value_function(np.atleast_2d(coalition))
    expected_sum = 1.0 + 0.8 + 0.5
    np.testing.assert_allclose(result_value_function, expected_sum, atol=0.05)
