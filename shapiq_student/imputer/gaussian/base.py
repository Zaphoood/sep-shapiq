"""Abstract base class for imputation approaches.

This module defines the base class for different Gaussian-based imputation approaches used in Shapley Value
calculations. It inherits from shapiq's Imputer base class and provides a common interface that all specific approaches must implement.
"""

# TODO(Zaphoood): Make sure all docstrings follow Google's style

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng

from .exceptions import CategoricalFeatureError, EmptyDataError

if TYPE_CHECKING:
    import numpy.typing as npt

from shapiq.games.imputer.base import Imputer

# We disallow columns with <= 2 unique values, since they are likely either:
# - Binary features
# - One-hot encoded features (which would have at most 2 values per encoded column)
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 2


class GaussianImputerBase(Imputer):
    """Abstract base class for Gaussian-based imputation approaches.

    This class inherits from shapiq's Imputer base class and defines the interface that all specific
    Gaussian-based imputation approaches must implement.
    """

    def __init__(
        self,
        model: object,
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        n_mc_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """Initializes GaussianImputerBase.

        Args:
            model (object): The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data (npt.NDArray[np.floating]): The background data to use for the explainer as a 2-dimensional array with shape
                (n_samples, n_features).
            x (npt.NDArray[np.floating] | None, optional): The explanation point to use the imputer on either as a 2-dimensional array with
                shape (1, n_features) or as a vector with shape (n_features,). Defaults to None.
            n_mc_samples (int, optional): Number of Monte Carlo samples for imputation. Defaults to 1000.
            random_state (int | None, optional): The random state to use for sampling. Defaults to None.

        Raises:
            EmptyDataError: If the provided data is empty.
        """
        if data is None or np.size(data) == 0 or (hasattr(data, "shape") and data.shape[0] == 0):
            raise EmptyDataError
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=n_mc_samples,
            categorical_features=[],
            random_state=random_state,
        )

        self.n_mc_samples = n_mc_samples
        self._mean_per_feature: npt.NDArray[np.floating] | None = None
        self._cov_mat: npt.NDArray[np.floating] | None = None

    def _check_categorical_features(self) -> None:
        """Check if any features are categorical variables.

        Raises:
            CategoricalFeatureError: If any categorical features are detected.
        """
        categorical_indices: list[int] = []
        for i, col in enumerate(self.data.T):
            if any(isinstance(v, str) for v in col):
                categorical_indices.append(i)
                continue
            unique_count = len(np.unique(col))
            if unique_count <= MAX_UNIQUE_VALUES_FOR_CATEGORICAL:
                categorical_indices.append(i)
        if categorical_indices:
            raise CategoricalFeatureError(categorical_indices)

    @property
    def mean_per_feature(self) -> npt.NDArray[np.floating]:
        """Get the mean values per feature, computing them if not already computed.

        Returns:
            npt.NDArray[np.floating]: The mean value for each feature.
        """
        if self._mean_per_feature is None:
            self._mean_per_feature = np.mean(self.data, axis=0)
        return self._mean_per_feature

    @property
    def cov_mat(self) -> npt.NDArray[np.floating]:
        """Get the covariance matrix, computing it if not already computed.

        Returns:
            npt.NDArray[np.floating]: The covariance matrix of the data.
        """
        if self._cov_mat is None:
            self._cov_mat = self._ensure_positive_definite(np.cov(self.data.T))
        return self._cov_mat

    def _ensure_positive_definite(
        self,
        cov_mat: npt.NDArray[np.floating],
        min_eigen_value: float = 1e-06,
    ) -> npt.NDArray[np.floating]:
        """Ensure covariance matrix is positive definite by correcting eigenvalues if necessary.

        Args:
            cov_mat: Input covariance matrix.
            min_eigen_value: Minimum allowed eigenvalue. Defaults to 1e-06.

        Returns:
            npt.NDArray[np.floating]: Positive definite covariance matrix.
        """
        eigen_values = np.linalg.eigvalsh(cov_mat)

        # If any eigenvalue is too small (close to zero or negative)
        if np.any(eigen_values <= min_eigen_value):
            # Add regularization to make it positive definite
            min_actual_eigen_value = np.min(eigen_values)
            cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * (
                min_eigen_value - min_actual_eigen_value
            )
        return cov_mat

    def impute(
        self,
        x: npt.NDArray[np.floating],
        coalitions: npt.NDArray[np.bool],
    ) -> npt.NDArray[np.floating]:
        """Impute missing values for given coalitions using Gaussian MC sampling.

        Args:
            coalitions: Binary array indicating which features are present (1) or missing (0)
                for each coalition. Shape: (n_coalitions, n_features).
            x: Explanation point to use for imputation. If None, uses self.x.

        Returns:
            Imputed data points for each coalition.
            Shape: (n_coalitions, n_mc_samples, n_features)
        """
        x_explain = x.flatten()
        n_coalitions, n_features = coalitions.shape
        rng = default_rng(self.random_state)

        imputed_data = np.zeros((n_coalitions, self.n_mc_samples, n_features))

        for S_ind, coalition in enumerate(coalitions):
            S_idx_known = np.where(coalition)[0]
            S_idx_unknown = np.where(~coalition)[0]

            if len(S_idx_known) == 0:
                # No conditioning on known features, therefore sample from original data distribution
                Z = rng.standard_normal((self.n_mc_samples, len(S_idx_unknown)))
                samples = Z @ np.linalg.cholesky(self.cov_mat).T + self.mean_per_feature
            elif len(S_idx_unknown) == 0:
                # all features known, therefore we just return explanation point
                samples = np.tile(x_explain, (self.n_mc_samples, 1))
                # TODO (milanagm): aus dem loop aussteigen und x_explain direkt unten in samples eintragen
            else:
                x_S_star = x_explain[S_idx_known]

                mu_S_known = self.mean_per_feature[S_idx_known]
                mu_S_unknown = self.mean_per_feature[S_idx_unknown]

                cov_S_known_known = self.cov_mat[np.ix_(S_idx_known, S_idx_known)]
                cov_S_known_unknown = self.cov_mat[np.ix_(S_idx_known, S_idx_unknown)]
                cov_S_unknown_known = self.cov_mat[np.ix_(S_idx_unknown, S_idx_known)]
                cov_S_unknown_unknown = self.cov_mat[np.ix_(S_idx_unknown, S_idx_unknown)]

                cov_S_known_known_inv = np.linalg.inv(cov_S_known_known)

                cond_mean = mu_S_unknown + (cov_S_unknown_known @ cov_S_known_known_inv) @ (
                    x_S_star - mu_S_known
                )
                cond_cov = (
                    cov_S_unknown_unknown
                    - (cov_S_unknown_known @ cov_S_known_known_inv) @ cov_S_known_unknown
                )
                # for sampling from multivariate normal distribution with Cholesky we need to make sure that
                # cond_cov is symmetric (regardless - Covariances should always be symmetric: Cov(X,Y) = Cov(Y,X))
                cond_cov = 0.5 * (cond_cov + cond_cov.T)

                # MC samples and Cholesky to turn N(0,1) to desired Gaussian distribution
                Z = rng.standard_normal((self.n_mc_samples, len(S_idx_unknown)))
                samples_unknown = Z @ np.linalg.cholesky(cond_cov).T + cond_mean

                samples = np.tile(x_explain, (self.n_mc_samples, 1))
                samples[:, S_idx_unknown] = samples_unknown

            imputed_data[S_ind] = samples

        return imputed_data
