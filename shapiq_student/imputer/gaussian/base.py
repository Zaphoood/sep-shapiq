"""Abstract base class for imputation approaches.

This module defines the base class for different Gaussian-based imputation approaches used in Shapley Value
calculations. It inherits from shapiq's Imputer base class and provides a common interface that all specific approaches must implement.
"""

# TODO(Zaphoood): Make sure all docstrings follow Google's style

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, override

import numpy as np
from numpy.random import default_rng

from .exceptions import CategoricalFeatureError

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from shapiq import Game

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
        model: object | Game | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        n_mc_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """Initializes the class.

        Args:
            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data: The background data to use for the explainer as a ``np.ndarray`` of shape ``(n_samples, n_features)``.
            x: The explanation point as a ``np.ndarray`` of shape ``(1, n_features)`` or ``(n_features,)``. Defaults to ``None``.
            n_mc_samples: Number of Monte Carlo samples for imputation. Defaults to 1000.
            random_state: The random state to use for sampling. Defaults to ``None``.

        Raises:
            EmptyDataError: If the provided data is empty.
        """
        if data is None or np.size(data) == 0 or (hasattr(data, "shape") and data.shape[0] == 0):
            msg = "Training data is empty."
            raise ValueError(msg)

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
            An array containing the mean value for each feature.
        """
        if self._mean_per_feature is None:
            self._mean_per_feature = np.mean(self.data, axis=0)
        return self._mean_per_feature

    @property
    def cov_mat(self) -> npt.NDArray[np.floating]:
        """Compute the covariance matrix or return a cached value if already computed.

        Returns:
            The covariance matrix of the data as an array.
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
            The positive definite covariance matrix.
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

    def sample_monte_carlo(
        self,
        x: npt.NDArray[np.floating],
        coalitions: npt.NDArray[np.bool],
    ) -> npt.NDArray[np.floating]:
        """GEnerate Gaussian Monte Carlo samples for missing values for given coalitions.

        Args:
            coalitions: Binary array of shape ``(n_coalitions, n_features)`` indicating which features are present (1) or missing (0)
                for each coalition.
            x: Explanation point to use for imputation. If ``None``, uses ``self.x``.

        Returns:
            Imputed data points for each coalition as an array of shape ``(n_coalitions, n_mc_samples, n_features)``.
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

    @abstractmethod
    def impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Impute missing values for given coalitions. This method must be override by each subclass.

        Args:
            x: The data point to impute as an array of shape ``(n_features,)``.
            coalitions: Binary array of shape ``(n_coalitions, n_features)`` indicating which features are present (1) or missing (0) for each coalition.

        Returns:
            An array of shape ``(n_coalitions, n_features)`` containing the imputed data points for each coalition.

        Raises:
            RuntimeError: If no explanation point has been provided, neither in the constructor nor by calling ``fit()``.
        """
        msg = f"The impute() method must be implemented by each subclass of {self.__class__.__name__}."
        raise NotImplementedError(msg)

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Impute missing values and return model predictions for given coalitions.

        This method performs imputation and then calls the model's prediction function
        on the imputed data. This is the main interface expected by Shapley Value explainers.

        Args:
            coalitions: Binary array of shape ``(n_coalitions, n_features)`` indicating which features are present (1) or missing (0)
                for each coalition.

        Returns:
            An array of shape ``(n_coalitions,)`` with model predictions for each imputed data point.

        Raises:
            RuntimeError: If no explanation point has been provided, neither in the constructor nor by calling ``fit()``.
        """
        if self.x is None:
            msg = f"Must call {self.__class__.__name__}.fit() first before imputing"
            raise RuntimeError(msg)

        return self.predict(self.impute(self.x, coalitions))
