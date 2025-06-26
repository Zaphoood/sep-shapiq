"""Gaussian approach for imputation in SHAP value calculations.

This module implements a Gaussian-based approach for generating samples in SHAP value
calculations. It uses multivariate normal distribution for sampling and handles
continuous features only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import Generator, default_rng

from .approach import Approach

if TYPE_CHECKING:
    from numpy.typing import NDArray

# We disallow columns with <= 2 unique values, since they are likely either:
# - Binary features
# - One-hot encoded features (which would have at most 2 values per encoded column)
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 2
# Threshold for coalition membership (0.5 separates 0 and 1 in binary representation)
COALITION_THRESHOLD = 0.5


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
            "Gaussian approach does not support categorical features."
        )
        super().__init__(message)


class GaussianApproach(Approach):
    """Implementation of Gaussian-based approach for SHAP value calculations.

    This approach uses multivariate normal distribution for generating samples.
    It only supports continuous features and raises an error if categorical features
    are detected.
    """

    def _check_categorical_features(self) -> None:
        """Check if any features are categorical variables. This method just needs to be passed.

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

    def calculate_mean_per_feature(self) -> None:
        """Calculate the mean value for each feature in the training data.

        This method computes the mean value for each feature (column) in the training data
        and stores it in the internal parameters dictionary.

        Raises:
        ------
        TypeError
            If training data is not a numpy array.
        ValueError
            If training data is empty.
        """
        x_train = self.internal.get("data", {}).get("x_train")
        if not isinstance(x_train, np.ndarray):
            msg = "Training data must be a numpy array."
            raise TypeError(msg)
        if x_train.size == 0:
            msg = "Training data is empty."
            raise ValueError(msg)
        if "mean_per_feature" not in self.internal["parameters"]:
            self.internal["parameters"]["mean_per_feature"] = np.mean(x_train, axis=0)

    def _ensure_positive_definite(
        self, cov_mat: NDArray[np.float64], min_eigen_value: float = 1e-06
    ) -> NDArray[np.float64]:
        """Ensure covariance matrix is positive definite by correcting eigenvalues if necessary.

        Parameters
        ----------
        cov_mat : np.ndarray
            Input covariance matrix
        min_eigen_value : float, optional
            Minimum allowed eigenvalue, by default 1e-06

        Returns:
        -------
        np.ndarray
            Positive definite covariance matrix
        """
        eigen_values = np.linalg.eigvalsh(cov_mat)

        # If any eigenvalue is too small (close to zero or negative)
        if np.any(eigen_values <= min_eigen_value):
            # Add regularization to make it positive definite
            min_eig = np.min(eigen_values)
            cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * (min_eigen_value - min_eig)

        return cov_mat

    def calculate_covariance_matrix(self) -> None:
        """Compute the covariance matrix of the training data.

        This method calculates the covariance matrix from the training data
        and stores it in the internal parameters dictionary.

        Raises:
        ------
        TypeError
            If training data is not a numpy array.
        ValueError
            If training data is empty.
        """
        x_train = self.internal.get("data", {}).get("x_train")
        if not isinstance(x_train, np.ndarray):
            msg = "Training data must be a numpy array."
            raise TypeError(msg)
        if x_train.size == 0:
            msg = "Training data is empty."
            raise ValueError(msg)
        if "cov_mat" not in self.internal["parameters"]:
            cov_mat = np.cov(x_train.T)
            # Ensure positive definiteness
            cov_mat = self._ensure_positive_definite(cov_mat)
            self.internal["parameters"]["cov_mat"] = cov_mat

    def __init__(self, internal: dict[str, Any]) -> None:
        """Initialize the GaussianApproach with internal parameters and perform categorical feature check.

        Parameters
        ----------
        internal : dict[str, Any]
            Internal dictionary containing data and parameters for the approach.
        """
        super().__init__(internal)

        # Categorical-Check
        self._check_categorical_features()

        # Ensure mean and covariance are initialized via helper methods
        self.calculate_mean_per_feature()
        self.calculate_covariance_matrix()

    def gaussian_imputation(self) -> NDArray[np.float64]:
        """Perform Gaussian imputation for SHAP value calculations.

        This method generates samples from a multivariate normal distribution
        based on the training data and calculates the SHAP values for each feature.

        Returns:
        -------
        np.ndarray
            Resulting SHAP values for each feature.
        """
        n_features = self.internal["parameters"]["n_features"]
        n_MC_samples = self.internal["parameters"]["n_MC_samples"]
        x_explain_mat = np.array(self.internal["data"]["x_explain"])
        n_explain = self.internal["parameters"]["n_explain"]
        mean_per_feature = self.internal["parameters"]["mean_per_feature"]
        cov_mat = self.internal["parameters"]["cov_mat"]

        # Calculating Coalition Matrix S
        n_coalitions = 2**n_features
        S = np.zeros((n_coalitions, n_features), dtype=int)
        for i in range(n_coalitions):
            # Binary representation: 1 = feature present in coalition, 0 = not present
            S[i, :] = [(i >> j) & 1 for j in range(n_features - 1, -1, -1)]
        # S now has shape (n_coalitions, n_features), each row is a coalition

        # Calculating MC Samples
        rng: Generator = default_rng()
        # Draw directly from multivariate normal: samples ~ N(mu, cov)
        MC_samples_mat = rng.multivariate_normal(
            mean=mean_per_feature, cov=cov_mat, size=n_MC_samples
        )

        # Result array: [n_MC_samples, n_explain * n_coalitions, n_features]
        result_cube = np.zeros((n_MC_samples, n_explain * n_coalitions, n_features))

        for S_ind in range(n_coalitions):
            # Iterate over all rows of S
            # Imputing for every coalition

            S_now = S[S_ind, :]  # current coalition
            S_now_idx_known = np.where(S_now > COALITION_THRESHOLD)[0]  # Indices of known features
            S_now_idx_unkown = np.where(S_now < COALITION_THRESHOLD)[
                0
            ]  # Indices of features to impute

            # Means and covariance submatrices
            mu_S_known = mean_per_feature[S_now_idx_known]
            mu_S_unknown = mean_per_feature[S_now_idx_unkown]

            cov_S_known_and_S_known = cov_mat[np.ix_(S_now_idx_known, S_now_idx_known)]
            cov_S_known_and_S_unknown = cov_mat[np.ix_(S_now_idx_known, S_now_idx_unkown)]

            cov_S_unknown_and_S_known = cov_mat[np.ix_(S_now_idx_unkown, S_now_idx_known)]
            cov_S_unknown_and_S_unknown = cov_mat[np.ix_(S_now_idx_unkown, S_now_idx_unkown)]

            # Conditional covariance and mean
            cov_SS_inv = (
                np.linalg.inv(cov_S_known_and_S_known)
                if cov_S_known_and_S_known.size
                else np.zeros((0, 0))
            )
            cov_SbarS_cov_SS_inv = (
                cov_S_unknown_and_S_known @ cov_SS_inv
                if cov_S_known_and_S_known.size
                else np.zeros((len(S_now_idx_unkown), 0))
            )
            cond_cov = (
                cov_S_unknown_and_S_unknown - cov_SbarS_cov_SS_inv @ cov_S_known_and_S_unknown
                if cov_S_known_and_S_known.size
                else cov_S_unknown_and_S_unknown
            )

            # Cholesky decomposition for imputation
            chol_cond_cov = np.linalg.cholesky(cond_cov) if cond_cov.size else np.zeros((0, 0))

            for idx_now in range(n_explain):
                # Known values for this observation
                x_S_star = (
                    x_explain_mat[idx_now, S_now_idx_known]
                    if S_now_idx_known.size
                    else np.array([])
                )

                # Conditional mean for this observation
                if S_now_idx_known.size:
                    x_Sbar_mean = cov_SbarS_cov_SS_inv @ (x_S_star - mu_S_known) + mu_S_unknown
                else:
                    x_Sbar_mean = mu_S_unknown

                # Imputed values for the unknown features
                if S_now_idx_unkown.size:
                    MC_samples_now = (
                        MC_samples_mat[:, S_now_idx_unkown] @ chol_cond_cov + x_Sbar_mean
                    )
                else:
                    MC_samples_now = np.zeros((n_MC_samples, 0))

                # Helper array for all features
                aux_mat = np.zeros((n_MC_samples, n_features))
                if S_now_idx_known.size:
                    aux_mat[:, S_now_idx_known] = x_S_star  # known features: always the same
                if S_now_idx_unkown.size:
                    aux_mat[:, S_now_idx_unkown] = MC_samples_now  # imputed features

                # Store in result
                result_cube[:, S_ind * n_explain + idx_now, :] = aux_mat

        return result_cube
