"""Tests for the GaussianImputer class."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pytest

from shapiq_student.imputer.gaussian.exceptions import CategoricalFeatureError, EmptyDataError
from shapiq_student.imputer.gaussian.gaussian_imputer import GaussianImputer


def dummy_model(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """A simple placeholder model for testing that returns the sum over the last axis."""
    return np.asarray(np.sum(x, axis=-1), dtype=float)


def test_check_categorical_features_valid() -> None:
    """Should pass silently when all features are continuous with >2 uniques."""
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


def test_check_categorical_features_binary_integer() -> None:
    """Should raise ValueError naming column f2 when it has only 0/1 values."""
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


def test_check_categorical_features_string() -> None:
    """Should raise ValueError naming column f2 when it contains strings."""
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


def test_check_categorical_features_mixed() -> None:
    """Should raise ValueError naming columns f2 and f3 for binary+string mix."""
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
    """Test mean calculation with empty data."""
    data = np.empty((0, 3))
    x = np.array([])
    with pytest.raises(EmptyDataError):
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
    """Test covariance calculation with empty data."""
    data = np.empty((0, 3))
    x = np.array([])
    with pytest.raises(EmptyDataError):
        GaussianImputer(model=dummy_model, data=data, x=x)


##############################################
# Test for Imputation ---                    #
##############################################


def test_gaussian_imputation_first_feature_known_mean_and_cov_check() -> None:
    """Test imputation: first feature known (1.0), last two unknown, mean should be [0.4, 0.25]."""
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
    """Test imputation: first feature known, last two set to 1, mean should be 0.4."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    rng = np.random.default_rng()
    x_train = rng.multivariate_normal(mean, cov, size=10000)
    x_explain = np.array([[1.0, np.nan, np.nan]])
    n_features = x_train.shape[1]
    # ususally coalitions (=S) are ot callculated within the imputer in shapiq package
    coalitions = np.array(list(product([0, 1], repeat=n_features)))

    imputer = GaussianImputer(
        model=dummy_model,
        data=x_train,
        x=x_explain[0],
    )
    result_cube = imputer.impute(coalitions)  # shape: (1, n_coalitions, n_mc_samples, n_features)

    # Find the coalition index for S = [1, 0, 0]
    n_coalitions = 2**n_features
    S = np.zeros((n_coalitions, n_features), dtype=int)
    for i in range(2**n_features):
        S[i, :] = [(i >> j) & 1 for j in range(n_features - 1, -1, -1)]
    coalition_idx = np.where((S == [1, 0, 0]).all(axis=1))[0][0]

    # For our explicand (idx 0), coalition S = [1,0,0] is at index: [0, coalition_idx, :, :]
    imputed_last_two = result_cube[0, coalition_idx, :, 1:3]  # all samples, features 1 and 2

    # The mean should be close to [0.8, 0.5]
    imputed_mean = np.mean(imputed_last_two, axis=0)
    np.testing.assert_allclose(imputed_mean, [0.8, 0.5], atol=0.05)
