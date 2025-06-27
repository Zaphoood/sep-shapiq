"""Tests for the GaussianApproach class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from shapiq_student.imputer.gaussian_approach import CategoricalFeatureError, GaussianApproach
from shapiq_student.imputer.main import impute


def test_check_categorical_features_valid() -> None:
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


def test_check_categorical_features_binary_integer() -> None:
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


def test_check_categorical_features_string() -> None:
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


def test_check_categorical_features_mixed() -> None:
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


##############################################
# Tests for Cov Mat and Mean Calculation --- #
##############################################


def test_calculate_mean_per_feature_valid() -> None:
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
    # Fix for mypy: cast to np.ndarray
    x_train_arr = np.asarray(cast(dict[str, Any], internal)["data"]["x_train"])
    expected_mean = np.mean(x_train_arr, axis=0)
    approach.calculate_mean_per_feature()
    calculated_mean: NDArray[np.float64] = np.asarray(
        approach.internal["parameters"]["mean_per_feature"], dtype=np.float64
    )
    assert "mean_per_feature" in approach.internal["parameters"]
    np.testing.assert_array_almost_equal(calculated_mean, expected_mean)
    # Check if mean has correct shape
    assert calculated_mean.shape == (3,)


def test_calculate_mean_per_feature_empty_data() -> None:
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


def test_calculate_mean_per_feature_invalid_data() -> None:
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


def test_calculate_covariance_matrix_valid() -> None:
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
    # Fix for mypy: cast to np.ndarray
    x_train_arr = np.asarray(cast(dict[str, Any], internal)["data"]["x_train"])
    expected_cov = np.cov(x_train_arr.T)
    approach.calculate_covariance_matrix()
    calculated_cov: NDArray[np.float64] = np.asarray(
        approach.internal["parameters"]["cov_mat"], dtype=np.float64
    )
    assert "cov_mat" in approach.internal["parameters"]
    np.testing.assert_array_almost_equal(calculated_cov, expected_cov)
    assert calculated_cov.shape == (3, 3)


def test_calculate_covariance_matrix_empty_data() -> None:
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


def test_calculate_covariance_matrix_invalid_data() -> None:
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


##############################################
# Test for Imputation --- #
##############################################


def test_gaussian_imputation_basic() -> None:
    """Test basic functionality of Gaussian imputation."""
    # Small synthetic dataset
    x_train = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    x_explain = np.array(
        [
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
        ]
    )
    n_features = x_train.shape[1]
    n_explain = x_explain.shape[0]
    n_MC_samples = 10

    result_cube = impute(
        x_train=x_train,
        x_explain=x_explain,
        approach="gaussian",
        feature_names=[f"f{i + 1}" for i in range(n_features)],
        n_MC_samples=10,
        verbose=["progress"],
    )

    # There are 2^n_features coalitions, n_explain explicands, n_MC_samples samples, n_features features
    assert result_cube is not None
    expected_shape = (n_MC_samples, n_explain * 2**n_features, n_features)
    assert result_cube.shape == expected_shape, (
        f"Shape mismatch: {result_cube.shape} != {expected_shape}"
    )
    # Verify the result has the expected properties
    assert result_cube.shape[0] == n_MC_samples
    assert result_cube.shape[2] == n_features


def test_gaussian_imputation_first_feature_known_mean_and_cov_check() -> None:
    """Test imputation: first feature known (1.0), last two unknown, mean should be [0.4, 0.25]."""
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    # Generate training data
    rng = np.random.default_rng()
    x_train = rng.random.multivariate_normal(mean, cov, size=10000)

    # Check sample mean and covariance
    sample_mean = np.mean(x_train, axis=0)
    sample_cov = np.cov(x_train, rowvar=False)
    np.testing.assert_allclose(sample_mean, mean, atol=0.05)
    np.testing.assert_allclose(sample_cov, cov, atol=0.05)


def test_gaussian_imputation_first_feature_known() -> None:
    """Test imputation: first feature known, last two set to 1, mean should be 0.4."""
    # dataset provided by supervisor
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]])
    # Generate training data
    rng = np.random.default_rng()
    x_train = rng.random.multivariate_normal(mean, cov, size=10000)

    # Condition: first feature unknown, last two known and set to 1
    x_explain = np.array([[1.0, np.nan, np.nan]])

    n_features = x_train.shape[1]
    n_explain = x_explain.shape[0]

    # Use your impute function
    result_cube = impute(
        x_train=x_train,
        x_explain=x_explain,
        approach="gaussian",
        feature_names=["f1", "f2", "f3"],
        n_MC_samples=1000,
        verbose=["progress"],
    )

    # Find the coalition index for S = [1, 0, 0]
    n_coalitions = 2**n_features
    S = np.zeros((n_coalitions, n_features), dtype=int)
    for i in range(2**n_features):
        S[i, :] = [(i >> j) & 1 for j in range(n_features - 1, -1, -1)]
    coalition_idx = np.where((S == [1, 0, 0]).all(axis=1))[0][0]

    # For our explicand (idx 0), coalition S = [1,0,0] is at index: coalition_idx * n_explain + 0
    idx = coalition_idx * n_explain + 0
    assert result_cube is not None
    imputed_last_two = result_cube[:, idx, 1:3]  # all samples, features 1 and 2

    # The mean should be close to [0.4, 0.25]
    imputed_mean = np.mean(imputed_last_two, axis=0)
    # np.testing.assert_allclose(imputed_mean, [0.4, 0.25], atol=0.05) # TODO (milanagm): I believe this is wrong # noqa: ERA001
    np.testing.assert_allclose(imputed_mean, [0.8, 0.5], atol=0.05)
