"""Tests for the GaussianApproach class."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_student.imputer.gaussian_approach import GaussianApproach


@pytest.fixture
def sample_data():
    """Create sample data for testing.

    Returns:
    -------
    np.ndarray
        A 3x3 array with sample data for testing mean and covariance calculations.
    """
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.fixture
def gaussian_approach(sample_data):
    """Create a GaussianApproach instance with sample data.

    Parameters
    ----------
    sample_data : np.ndarray
        The sample data to use for testing.

    Returns:
    -------
    GaussianApproach
        An initialized GaussianApproach instance with the sample data.
    """
    internal = {
        "data": {"x_train": sample_data},
        "parameters": {
            "feature_names": ["X1", "X2", "X3"],
            "n_features": 3,
            "verbose": [],
        },
    }
    return GaussianApproach(internal)


def test_calculate_mean_per_feature(gaussian_approach, sample_data):
    """Test mean calculation with valid data.

    This test verifies that:
    1. The mean is calculated correctly
    2. The result is stored in the correct location
    3. The calculation matches numpy's mean function
    """
    expected_mean = np.mean(sample_data, axis=0)
    gaussian_approach.calculate_mean_per_feature()
    calculated_mean = gaussian_approach.internal["parameters"]["mean_per_feature"]

    assert "mean_per_feature" in gaussian_approach.internal["parameters"]
    np.testing.assert_array_almost_equal(calculated_mean, expected_mean)
    assert calculated_mean.shape == (3,)


def test_calculate_mean_per_feature_empty_data():
    """Test mean calculation with empty data.

    This test verifies that the method raises a ValueError when given empty data.
    """
    internal = {
        "data": {"x_train": np.array([])},
        "parameters": {
            "feature_names": [],
            "n_features": 0,
            "verbose": [],
        },
    }
    approach = GaussianApproach(internal)
    with pytest.raises(ValueError, match="Training data is empty"):
        approach.calculate_mean_per_feature()


def test_calculate_mean_per_feature_invalid_data():
    """Test mean calculation with invalid data type.

    This test verifies that the method raises a TypeError when given non-numpy data.
    """
    internal = {
        "data": {"x_train": [1, 2, 3]},  # Not a numpy array
        "parameters": {
            "feature_names": ["X1"],
            "n_features": 1,
            "verbose": [],
        },
    }
    approach = GaussianApproach(internal)
    with pytest.raises(TypeError, match="Training data must be a numpy array"):
        approach.calculate_mean_per_feature()


def test_calculate_covariance_matrix(gaussian_approach, sample_data):
    """Test covariance matrix calculation with valid data.

    This test verifies that:
    1. The covariance matrix is calculated correctly
    2. The result is stored in the correct location
    3. The calculation matches numpy's cov function
    """
    expected_cov = np.cov(sample_data.T)
    gaussian_approach.calculate_covariance_matrix()
    calculated_cov = gaussian_approach.internal["parameters"]["cov_mat"]

    assert "cov_mat" in gaussian_approach.internal["parameters"]
    np.testing.assert_array_almost_equal(calculated_cov, expected_cov)
    assert calculated_cov.shape == (3, 3)


def test_calculate_covariance_matrix_empty_data():
    """Test covariance calculation with empty data.

    This test verifies that the method raises a ValueError when given empty data.
    """
    internal = {
        "data": {"x_train": np.array([])},
        "parameters": {
            "feature_names": [],
            "n_features": 0,
            "verbose": [],
        },
    }
    approach = GaussianApproach(internal)
    with pytest.raises(ValueError, match="Training data is empty"):
        approach.calculate_covariance_matrix()


def test_calculate_covariance_matrix_invalid_data():
    """Test covariance calculation with invalid data type.

    This test verifies that the method raises a TypeError when given non-numpy data.
    """
    internal = {
        "data": {"x_train": [1, 2, 3]},  # Not a numpy array
        "parameters": {
            "feature_names": ["X1"],
            "n_features": 1,
            "verbose": [],
        },
    }
    approach = GaussianApproach(internal)
    with pytest.raises(TypeError, match="Training data must be a numpy array"):
        approach.calculate_covariance_matrix()
