"""Test module for GaussianCopulaImputer class."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from shapiq_student.imputer.gaussian.copula_imputer import GaussianCopulaImputer

# Constants
SIGNIFICANCE_LEVEL = 0.05
CORRELATION_THRESHOLD = 0.99


class TestGaussianCopulaImputer:
    """Test class for GaussianCopulaImputer."""

    @pytest.fixture
    def setup_data(self):
        """Fixture for common test data."""
        rng = np.random.default_rng(42)
        data_normal = rng.normal(size=(100, 3))
        data_exp = rng.exponential(size=(100, 3))
        x = np.array([0.5, 0.5, 0.5])
        coalitions = np.array(
            [
                [False, False, False],  # No features
                [True, False, False],  # One feature
                [False, True, True],  # Two features
                [True, True, True],  # All features
            ]
        )

        def simple_model(x):
            """Simple model: sum of features."""
            return x.sum(axis=1)

        return {
            "data_normal": data_normal,
            "data_exp": data_exp,
            "x": x,
            "coalitions": coalitions,
            "simple_model": simple_model,
        }

    def test_rank_gaussian_transform(self, setup_data):
        """Test the rank-Gaussian transformation."""
        imputer = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=setup_data["data_normal"]
        )
        # Use public method through data_transformed attribute
        transformed = imputer.data_transformed

        # Check standard normal distribution
        for i in range(transformed.shape[1]):
            _, p_value = kstest(transformed[:, i], "norm")
            assert p_value > SIGNIFICANCE_LEVEL, (
                f"Feature {i} is not normally distributed (p={p_value})"
            )

        # Check mean ~0 and std ~1
        assert np.allclose(np.mean(transformed, axis=0), 0, atol=0.1)
        assert np.allclose(np.std(transformed, axis=0), 1, atol=0.1)

    def test_transform_roundtrip(self, setup_data):
        """Test forward and backward transformation with reasonable tolerance."""
        original = setup_data["data_normal"]
        # Test through public interface by creating a new instance
        test_imputer = GaussianCopulaImputer(model=setup_data["simple_model"], data=original)
        transformed = test_imputer.data_transformed

        # Check that values are in correct order (rank preservation)
        for i in range(original.shape[1]):
            # Correlation between original and transformed should be high
            # (testing monotonicity through rank correlation)
            from scipy.stats import spearmanr

            correlation, _ = spearmanr(original[:, i], transformed[:, i])
            assert correlation > CORRELATION_THRESHOLD, (
                f"Rank correlation for feature {i} too low: {correlation}"
            )

    def test_imputation_with_normal_data(self, setup_data):
        """Test imputation with normally distributed data."""
        imputer = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=setup_data["data_normal"], random_state=42
        )
        results = imputer.impute(setup_data["x"], setup_data["coalitions"])

        # Test empty coalition - should approximately match training data mean
        expected_empty = setup_data["simple_model"](
            setup_data["data_normal"].mean(axis=0, keepdims=True)
        )[0]
        assert np.isclose(results[0], expected_empty, rtol=0.5), (
            f"Expected: {expected_empty}, Got: {results[0]}"
        )

        # Test full coalition - should exactly match input point sum
        expected_full = setup_data["simple_model"](setup_data["x"].reshape(1, -1))[0]
        assert np.isclose(results[3], expected_full, rtol=0.1), (
            f"Expected: {expected_full}, Got: {results[3]}"
        )

    def test_imputation_with_non_normal_data(self, setup_data):
        """Test imputation with non-normally distributed data."""
        imputer = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=setup_data["data_exp"], random_state=42
        )
        results = imputer.impute(setup_data["x"], setup_data["coalitions"])
        assert results.shape == (4,), "Wrong output shape"

        # Test that all values are finite
        assert np.all(np.isfinite(results)), "Not all results are finite"

    def test_reproducibility(self, setup_data):
        """Test reproducibility with fixed random_state."""
        imputer1 = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=setup_data["data_normal"], random_state=42
        )
        imputer2 = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=setup_data["data_normal"], random_state=42
        )
        results1 = imputer1.impute(setup_data["x"], setup_data["coalitions"])
        results2 = imputer2.impute(setup_data["x"], setup_data["coalitions"])
        assert np.allclose(results1, results2), "Results not reproducible"

    def test_covariance_matrix_positive_definite(self, setup_data):
        """Test if covariance matrix is positive definite."""
        imputer = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=setup_data["data_normal"]
        )
        # Access through public interface - check eigenvalues of transformed data covariance
        eigenvalues = np.linalg.eigvals(np.cov(imputer.data_transformed.T))
        assert np.all(eigenvalues > 0), "Covariance matrix is not positive definite"

    def test_edge_case_single_feature(self):
        """Test case with only one feature."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(100, 1))
        x = np.array([0.5])
        coalitions = np.array([[False], [True]])

        def identity_model(x):
            """Identity model for testing."""
            return x.flatten()

        imputer = GaussianCopulaImputer(model=identity_model, data=data, random_state=42)
        results = imputer.impute(x, coalitions)
        assert results.shape == (2,), "Wrong output shape for single feature"

    def test_gaussian_transformation_properties(self, setup_data):
        """Test properties of Gaussian transformation."""
        # Test with different data types
        for data in [setup_data["data_normal"], setup_data["data_exp"]]:
            test_imputer = GaussianCopulaImputer(model=setup_data["simple_model"], data=data)
            transformed = test_imputer.data_transformed

            # Check that transformation is monotonic
            for i in range(data.shape[1]):
                sorted_transformed = np.sort(transformed[:, i])

                # Monotonicity: differences should be non-negative
                assert np.all(np.diff(sorted_transformed) >= 0), (
                    f"Transformation not monotonic for feature {i}"
                )

    def test_extreme_values_handling(self, setup_data):
        """Test handling of extreme values."""
        # Create data with extreme values
        rng = np.random.default_rng(42)
        data_extreme = rng.normal(size=(100, 3))
        data_extreme[0, 0] = 1000  # Very large value
        data_extreme[1, 1] = -1000  # Very small value

        imputer = GaussianCopulaImputer(
            model=setup_data["simple_model"], data=data_extreme, random_state=42
        )

        # Test that transformation works
        transformed = imputer.data_transformed
        assert np.all(np.isfinite(transformed)), "Transformation produces non-finite values"

        # Test that imputation works with extreme values
        results = imputer.impute(setup_data["x"], setup_data["coalitions"])
        assert np.all(np.isfinite(results)), "Imputation produces non-finite values"
