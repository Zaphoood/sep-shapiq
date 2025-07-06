"""Copula's approach for imputation in SHAP value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import Generator, default_rng
from scipy.linalg import cholesky
from scipy.stats import norm, rankdata

from .approach import Approach

if TYPE_CHECKING:
    import numpy.typing as npt

# Constants
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 2
MIN_SAMPLES_FOR_COVARIANCE = 2


class CategoricalFeatureError(ValueError):
    """Exception raised when categorical features are detected."""

    def __init__(self, feature_names: list[str]) -> None:
        """Initialize the error with the categorical feature names.

        Parameters
        ----------
        feature_names : list[str]
            List of feature names that are categorical
        """
        self.feature_names = feature_names
        message = (
            f"The following are categorical features: {', '.join(feature_names)}. "
            "Gaussian Copula approach does not support categorical features."
        )
        super().__init__(message)


class CopulaApproach(Approach):
    """Implements the Gaussian Copula approach for feature imputation in SHAP value calculations."""

    def _check_categorical_features(self) -> None:
        """Check if any features are categorical variables.

        Raises:
        ------
        CategoricalFeatureError
            If any categorical features are detected.
        """
        x_train = self.internal["data"]["x_train"]
        feature_names = self.internal["parameters"]["feature_names"]

        categorical_features: list[str] = []
        for i, col in enumerate(x_train.T):
            if any(isinstance(v, str) for v in col):
                categorical_features.append(feature_names[i])
                continue

            unique_count = len(np.unique(col))
            if unique_count <= MAX_UNIQUE_VALUES_FOR_CATEGORICAL:
                categorical_features.append(feature_names[i])

        if categorical_features:
            raise CategoricalFeatureError(categorical_features)

    def _gaussian_transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform data to standard normal distribution using empirical CDF."""
        ranks = rankdata(x, method="average")
        u = ranks / (len(ranks) + 1)
        return norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

    def calculate_mean_per_feature(self) -> None:
        """Calculate and store the mean of the Gaussian-transformed training data."""
        x_train = self.internal["data"]["x_train"]
        x_train_gaussian = np.apply_along_axis(self._gaussian_transform, 0, x_train)
        self.internal["data"]["x_train_gaussian"] = x_train_gaussian
        self.internal["parameters"]["mean_per_feature"] = np.mean(x_train_gaussian, axis=0)

    def _ensure_positive_definite(
        self, cov_mat: npt.NDArray[np.floating], min_eigen_value: float = 1e-06
    ) -> npt.NDArray[np.floating]:
        eigen_values = np.linalg.eigvalsh(cov_mat)
        if np.any(eigen_values <= min_eigen_value):
            min_actual = np.min(eigen_values)
            cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * (min_eigen_value - min_actual)
        return cov_mat

    def calculate_covariance_matrix(self) -> None:
        """Compute and store the covariance matrix of Gaussian-transformed data."""
        x_train_gaussian = self.internal["data"]["x_train_gaussian"]
        cov_mat = np.cov(x_train_gaussian, rowvar=False)
        cov_mat = self._ensure_positive_definite(cov_mat)
        self.internal["parameters"]["cov_mat"] = cov_mat

    def __init__(self, internal: dict[str, Any]) -> None:
        """Initialize the CopulaApproach.

        Parameters:
        ----------
        internal : dict[str, Any]
            Internal dictionary containing data and parameters
        """
        super().__init__(internal)
        self._check_categorical_features()
        self.calculate_mean_per_feature()
        self.calculate_covariance_matrix()

    def _transform_explain_data(self) -> None:
        """Transform explanation data to Gaussian space using training data's ECDF."""
        x_train = self.internal["data"]["x_train"]
        x_explain = self.internal["data"]["x_explain"]
        x_explain_gaussian = np.zeros_like(x_explain)

        for i in range(x_explain.shape[1]):
            # Combine train and explain for ranking
            combined = np.concatenate([x_explain[:, i], x_train[:, i]])
            ranks = rankdata(combined, method="average")
            explain_ranks = ranks[: x_explain.shape[0]]

            # Adjust ranks to avoid ties
            adjusted_ranks = explain_ranks - rankdata(explain_ranks, method="average") + 0.5
            u = adjusted_ranks / (len(combined) - x_explain.shape[0] + 1)
            x_explain_gaussian[:, i] = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

        self.internal["data"]["x_explain_gaussian"] = x_explain_gaussian

    def _inverse_transform(self, z_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original space."""
        x_train = self.internal["data"]["x_train"]
        x_original = np.zeros_like(z_samples)

        for i in range(z_samples.shape[1]):
            sorted_train = np.sort(x_train[:, i])
            u = norm.cdf(z_samples[:, i])
            ranks = u * (len(sorted_train) - 1)
            idx_low = np.floor(ranks).astype(int)
            idx_high = np.minimum(idx_low + 1, len(sorted_train) - 1)
            frac = ranks - idx_low
            x_original[:, i] = (1 - frac) * sorted_train[idx_low] + frac * sorted_train[idx_high]

        return x_original

    def copula_imputation(self) -> npt.NDArray[np.floating]:
        """Perform Gaussian Copula imputation for SHAP value calculations.

        Returns:
        -------
        np.ndarray
            A 3D array of shape (n_MC_samples, n_explain * n_coalitions, n_features)
            containing all imputed samples in original feature space.
        """
        self._transform_explain_data()

        n_features = self.internal["parameters"]["n_features"]
        n_MC_samples = self.internal["parameters"]["n_MC_samples"]
        n_explain = self.internal["parameters"]["n_explain"]
        x_explain_gaussian = self.internal["data"]["x_explain_gaussian"]
        cov_mat = self.internal["parameters"]["cov_mat"]
        rng: Generator = default_rng()

        # Generate all possible coalitions
        n_coalitions = 2**n_features
        S = np.zeros((n_coalitions, n_features), dtype=bool)
        for i in range(n_coalitions):
            S[i, :] = [(i >> j) & 1 for j in range(n_features - 1, -1, -1)]

        # Result array
        result_cube = np.zeros((n_MC_samples, n_explain * n_coalitions, n_features))

        for S_ind in range(n_coalitions):
            for idx_now in range(n_explain):
                z_row = x_explain_gaussian[idx_now]
                coalition_mask = S[S_ind, :]
                S_now_idx_known = np.where(coalition_mask)[0]
                S_now_idx_unknown = np.where(~coalition_mask)[0]

                if len(S_now_idx_known) == 0:
                    # No conditioning - sample from marginal
                    Z = rng.standard_normal((n_MC_samples, n_features))
                    samples_gaussian = Z @ cholesky(cov_mat).T
                elif len(S_now_idx_unknown) == 0:
                    # All features known - just repeat the explicand
                    samples_gaussian = np.tile(z_row, (n_MC_samples, 1))
                else:
                    # Conditional sampling
                    cov_SS = cov_mat[np.ix_(S_now_idx_known, S_now_idx_known)]
                    cov_SbarS = cov_mat[np.ix_(S_now_idx_unknown, S_now_idx_known)]
                    cov_SbarSbar = cov_mat[np.ix_(S_now_idx_unknown, S_now_idx_unknown)]

                    cov_SS_inv = np.linalg.inv(cov_SS)
                    mu_cond = cov_SbarS @ cov_SS_inv @ z_row[S_now_idx_known]
                    cov_cond = cov_SbarSbar - cov_SbarS @ cov_SS_inv @ cov_SbarS.T

                    # Regularize if needed
                    try:
                        L_cond = cholesky(cov_cond)
                    except np.linalg.LinAlgError:
                        jitter = 1e-9 * np.eye(cov_cond.shape[0])
                        L_cond = cholesky(cov_cond + jitter)

                    epsilon = rng.standard_normal((n_MC_samples, len(S_now_idx_unknown)))
                    z_unknown = mu_cond + (L_cond @ epsilon.T).T

                    samples_gaussian = np.tile(z_row, (n_MC_samples, 1))
                    samples_gaussian[:, S_now_idx_unknown] = z_unknown

                # Transform back to original space
                result_cube[:, S_ind * n_explain + idx_now, :] = self._inverse_transform(
                    samples_gaussian
                )

        return result_cube
