"""Copula's approach for imputation in SHAP value calculations.

This module implements Copula's binning strategy for generating samples in SHAP value
calculations. It uses a binning approach to handle the distribution of features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.linalg import LinAlgError, cholesky, inv
from scipy.stats import norm, rankdata

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .approach import Approach
from .gaussian_approach import CategoricalFeatureError

# Constants for validation
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 2
MIN_SAMPLES_FOR_COVARIANCE = 2
MIN_N_Y_VALUE = 1

# Error messages
TRAINING_DATA_EMPTY_MSG = "Training data is empty"
EXPLANATION_DATA_EMPTY_MSG = "Explanation data is empty"
ITER_LIST_MISSING_MSG = "iter_list is missing or empty in internal state"
S_MISSING_MSG = "S is missing from the last iteration in iter_list"
INDEX_FEATURES_TYPE_MSG = "index_features must be a list"
INDEX_FEATURES_INVALID_MSG = "index_features contains invalid indices"
COVARIANCE_SAMPLES_MSG = "Need at least 2 samples to compute covariance matrix"
COVARIANCE_SQUARE_MSG = "Covariance matrix must be square"
NOT_IMPLEMENTED_MSG = "This method is not used in the current implementation"


class CopulaApproach(Approach):
    """Implementation of Copula's binning strategy for SHAP value calculations."""

    def __init__(self, internal: dict[str, Any]) -> None:
        """Initialize the CopulaApproach with internal parameters.

        Parameters
        ----------
        internal : dict[str, Any]
            Internal dictionary containing data and parameters for the approach.
        """
        super().__init__(internal)
        self._check_categorical_features()
        self._setup_copula_transformations()
        self._compute_and_store_covariance()

    def _check_categorical_features(self) -> None:
        """Ensure no categorical features are present."""
        x_train = self.internal["data"]["x_train"]
        feature_names = self.internal["parameters"]["feature_names"]

        categorical_features = []
        for i, col in enumerate(x_train.T):
            # Check for string values (explicit categorical)
            if any(isinstance(v, str) for v in col):
                categorical_features.append(feature_names[i])
                continue

            # Check for binary/low-cardinality numerical features
            unique_values = np.unique(col)
            if len(unique_values) <= MAX_UNIQUE_VALUES_FOR_CATEGORICAL:
                # Additional check
                if not np.issubdtype(col.dtype, np.number):
                    categorical_features.append(feature_names[i])
                elif len(unique_values) == MAX_UNIQUE_VALUES_FOR_CATEGORICAL and not set(
                    unique_values
                ).issubset({0, 1}):
                    # Only flag if not standard binary (0/1)
                    categorical_features.append(feature_names[i])

        if categorical_features:
            raise CategoricalFeatureError(categorical_features)

    def _setup_copula_transformations(self) -> None:
        """Transform training and explanation data to Gaussian space."""
        x_train = self.internal["data"]["x_train"]
        x_explain = self.internal["data"]["x_explain"]

        # Validate data
        if x_train.shape[1] != x_explain.shape[1]:
            msg = f"Feature dimension mismatch: x_train has {x_train.shape[1]} features, x_explain has {x_explain.shape[1]}"
            raise ValueError(msg)

        if x_train.shape[0] == 0:
            raise ValueError(TRAINING_DATA_EMPTY_MSG)

        if x_explain.shape[0] == 0:
            raise ValueError(EXPLANATION_DATA_EMPTY_MSG)

        # Transform training data
        x_train_gaussian = np.apply_along_axis(self._gaussian_transform, 0, x_train)
        self.internal["data"]["x_train_gaussian"] = x_train_gaussian

        # Transform explanation data
        n_explain = x_explain.shape[0]
        combined = np.vstack([x_explain, x_train])
        x_explain_gaussian = np.zeros_like(x_explain)

        for i in range(x_explain.shape[1]):
            col_data = combined[:, i]
            x_explain_gaussian[:, i] = self._gaussian_transform_separate(col_data, n_explain)

        self.internal["data"]["x_explain_gaussian"] = x_explain_gaussian

    def _compute_and_store_covariance(self) -> None:
        """Compute and store the covariance matrix of transformed training data."""
        cov_mat = self._compute_covariance()
        self.internal["parameters"]["copula.cov_mat"] = cov_mat

    def _compute_covariance(self) -> NDArray[np.float64]:
        """Compute covariance matrix of transformed training data."""
        x_train_gaussian = self.internal["data"]["x_train_gaussian"]

        if x_train_gaussian.shape[0] < MIN_SAMPLES_FOR_COVARIANCE:
            raise ValueError(COVARIANCE_SAMPLES_MSG)

        return np.cov(x_train_gaussian, rowvar=False)

    def _gaussian_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform data to standard normal distribution using empirical CDF."""
        n = len(x)
        ranks = rankdata(x, method="average")
        u = ranks / (n + 1)
        return norm.ppf(u)

    def _gaussian_transform_separate(
        self, yx: NDArray[np.float64], n_y: int
    ) -> NDArray[np.float64]:
        """Transform new data using existing empirical distribution."""
        if n_y <= 0 or n_y >= len(yx):
            msg = f"n_y ({n_y}) must be between {MIN_N_Y_VALUE} and {len(yx) - 1}"
            raise ValueError(msg)

        n_total = len(yx)
        ranks = rankdata(yx, method="average")
        y_ranks = ranks[:n_y]

        # Adjust ranks to avoid ties
        adjusted_ranks = y_ranks - rankdata(y_ranks, method="average") + 0.5
        u = adjusted_ranks / (n_total - n_y + 1)

        # Ensure u is within valid range for norm.ppf
        u = np.clip(u, 1e-10, 1 - 1e-10)
        return norm.ppf(u)

    def prepare_data(self, index_features: list[int] | None = None) -> dict[str, Any]:
        """Prepare conditional samples using Gaussian Copula approach."""
        # Extract parameters and data
        params = self.internal["parameters"]
        data = self.internal["data"]
        n_explain = params["n_explain"]
        n_features = params["n_features"]
        n_MC_samples = params["n_MC_samples"]

        # Validate that required data exists
        if "iter_list" not in self.internal or not self.internal["iter_list"]:
            raise ValueError(ITER_LIST_MISSING_MSG)

        if "S" not in self.internal["iter_list"][-1]:
            raise ValueError(S_MISSING_MSG)

        S = self.internal["iter_list"][-1]["S"]

        # Handle index_features
        if index_features is None:
            index_features = list(range(len(S)))
        else:
            # Validate index_features
            if not isinstance(index_features, list):
                raise TypeError(INDEX_FEATURES_TYPE_MSG)
            if any(i < 0 or i >= len(S) for i in index_features):
                raise ValueError(INDEX_FEATURES_INVALID_MSG)

        S_filtered = [S[i] for i in index_features]
        n_coalitions = len(S_filtered)

        # Generate base MC samples
        rng = np.random.default_rng()
        MC_samples = rng.normal(size=(n_MC_samples, n_features))

        # Prepare storage
        total_samples = n_coalitions * n_explain * n_MC_samples
        samples = np.zeros((total_samples, n_features))
        id_coalition = np.zeros(total_samples, dtype=int)
        ids = np.zeros(total_samples, dtype=int)

        # Generate conditional samples
        sample_idx = 0
        for coal_idx, coalition_mask in enumerate(S_filtered):
            for expl_idx in range(n_explain):
                # Get current explicand's Gaussian representation
                z_s = data["x_explain_gaussian"][expl_idx]

                # Generate conditional samples
                cond_samples = self._conditional_sample(coalition_mask, z_s, MC_samples)

                # Transform back to original space
                orig_samples = self._inverse_transform(cond_samples)

                # Store results
                chunk = slice(sample_idx, sample_idx + n_MC_samples)
                samples[chunk] = orig_samples
                id_coalition[chunk] = index_features[coal_idx]
                ids[chunk] = expl_idx
                sample_idx += n_MC_samples

        return {
            "samples": samples,
            "id_coalition": id_coalition,
            "id": ids,
            "w": np.full(total_samples, 1.0 / n_MC_samples),
        }

    def _generate_copula_samples(
        self,
        _x_explain: NDArray[np.float64],
        index_features: list[int] | None,
        _n_samples: int,
    ) -> NDArray[np.float64]:
        """Generate samples using Copula's approach."""
        if index_features is None:
            index_features = list(range(self.internal["parameters"]["copula.cov_mat"].shape[1]))
        # This method is not used in the current implementation
        # The actual sampling is done in prepare_data method
        raise NotImplementedError(NOT_IMPLEMENTED_MSG)

    def _conditional_sample(
        self,
        coalition_mask: NDArray[np.bool_],
        z_s: NDArray[np.float64],
        MC_samples: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Generate conditional samples in Gaussian space.

        Generates samples from the conditional distribution of unobserved features (S̄)
        given observed features (S) in Gaussian space, following the formula:
        z_S̄|S = μ_S̄|S + L_S̄|S * ε.
        """
        # Validate inputs
        if coalition_mask.shape[0] != z_s.shape[0]:
            msg = f"coalition_mask length ({coalition_mask.shape[0]}) must match z_s length ({z_s.shape[0]})"
            raise ValueError(msg)

        if MC_samples.shape[1] != z_s.shape[0]:
            msg = f"MC_samples feature dimension ({MC_samples.shape[1]}) must match z_s length ({z_s.shape[0]})"
            raise ValueError(msg)

        cov_mat = self.internal["parameters"]["copula.cov_mat"]

        if cov_mat.shape[0] != z_s.shape[0] or cov_mat.shape[1] != z_s.shape[0]:
            msg = f"Covariance matrix shape {cov_mat.shape} doesn't match feature dimension {z_s.shape[0]}"
            raise ValueError(msg)

        S_indices = np.where(coalition_mask)[0]
        Sbar_indices = np.where(~coalition_mask)[0]

        # Handle edge cases
        if len(S_indices) == 0:  # No conditioning features
            return self._marginal_sample(MC_samples, cov_mat)

        if len(Sbar_indices) == 0:  # All features observed
            return np.tile(z_s, (MC_samples.shape[0], 1))

        # Partition covariance matrix
        cov_SS = cov_mat[np.ix_(S_indices, S_indices)]
        cov_SbarS = cov_mat[np.ix_(Sbar_indices, S_indices)]
        cov_SbarSbar = cov_mat[np.ix_(Sbar_indices, Sbar_indices)]

        # Compute conditional parameters
        try:
            cov_SS_inv = inv(cov_SS)
        except LinAlgError:
            cov_SS_inv = np.linalg.pinv(cov_SS)

        mu_cond = cov_SbarS @ cov_SS_inv @ z_s[S_indices]
        cov_cond = cov_SbarSbar - cov_SbarS @ cov_SS_inv @ cov_SbarS.T

        # Regularize covariance if needed
        try:
            L_cond = cholesky(cov_cond)
        except LinAlgError:
            jitter = 1e-9 * np.eye(cov_cond.shape[0])
            L_cond = cholesky(cov_cond + jitter)

        # Generate conditional samples
        epsilon = MC_samples[:, Sbar_indices]
        z_sbar = mu_cond + (L_cond @ epsilon.T).T

        # Combine with observed features
        samples = np.tile(z_s, (MC_samples.shape[0], 1))
        samples[:, Sbar_indices] = z_sbar
        return samples

    def _marginal_sample(
        self, MC_samples: NDArray[np.float64], cov_mat: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Sample from marginal distribution."""
        # Validate inputs
        if cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError(COVARIANCE_SQUARE_MSG)

        if MC_samples.shape[1] != cov_mat.shape[0]:
            msg = f"MC_samples feature dimension ({MC_samples.shape[1]}) must match covariance matrix dimension ({cov_mat.shape[0]})"
            raise ValueError(msg)

        try:
            L = cholesky(cov_mat)
        except LinAlgError:
            jitter = 1e-9 * np.eye(cov_mat.shape[0])
            L = cholesky(cov_mat + jitter)
        return (L @ MC_samples.T).T

    def _inverse_transform(self, z_samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform Gaussian samples back to original feature space."""
        x_train = self.internal["data"]["x_train"]

        if x_train.shape[1] != z_samples.shape[1]:
            msg = f"Feature dimension mismatch: x_train has {x_train.shape[1]} features, z_samples has {z_samples.shape[1]}"
            raise ValueError(msg)

        x_original = np.zeros_like(z_samples)

        for i in range(z_samples.shape[1]):
            # Get empirical quantiles
            sorted_train = np.sort(x_train[:, i])
            n_train = len(sorted_train)

            if n_train == 0:
                msg = f"Training data is empty for feature {i}"
                raise ValueError(msg)

            # Compute quantiles
            u = norm.cdf(z_samples[:, i])
            # Ensure u is within valid range
            u = np.clip(u, 0, 1)
            ranks = u * (n_train - 1)
            idx_low = np.floor(ranks).astype(int)
            idx_high = np.minimum(idx_low + 1, n_train - 1)
            frac = ranks - idx_low

            # Linear interpolation
            x_original[:, i] = (1 - frac) * sorted_train[idx_low] + frac * sorted_train[idx_high]

        return x_original
