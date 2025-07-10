"""Copula's approach for imputation in Shapley Value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
from scipy.stats import norm, rankdata

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from shapiq import Game

from .base import GaussianImputerBase


class GaussianCopulaImputer(GaussianImputerBase):
    """Implements the Gaussian Copula approach for feature imputation in Shapley Value calculations."""

    @override
    def __init__(
        self,
        model: object | Game | Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        n_mc_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            model=model,
            data=data,
            x=x,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
        )
        self._check_categorical_features()
        self.data_transformed = self._gaussian_transform(self.data)
        # Override: GaussianCopulaImputer uses transformed mean/covariance
        self._mean_per_feature = np.mean(
            self.data_transformed, axis=0
        )  # in theory mean should be (nearly) zero
        self._cov_mat = self._ensure_positive_definite(np.cov(self.data_transformed.T))

    def _gaussian_transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform each feature to standard normal using empirical CDF (rank-Gaussian).

        For each feature (column), this method applies a transformation so that the values follow a standard normal
        distribution (mean 0, std 1), while preserving the rank order of the original data. This is also known as a
        rank-Gaussian or empirical CDF transformation.
        """
        x_transformed = np.zeros_like(x, dtype=float)
        for i in range(x.shape[1]):
            ranks = rankdata(x[:, i], method="average")
            quantile = ranks / (len(ranks) + 1)
            x_transformed[:, i] = norm.ppf(np.clip(quantile, 1e-10, 1 - 1e-10))
        return x_transformed

    def _transform_x_explain(
        self, x_explain: npt.NDArray[np.floating], x_train: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform a single explanation point to Gaussian space using the training data's ECDF.

        Args:
            x_explain: The explanation point with shape ``(n_features,)``
            x_train: The training data with shape ``(n_samples, n_features)``

        Returns:
            Transformed explanation point in Gaussian space as an array of shape ``(n_features,)``
        """
        x = x_explain
        x_train = np.asarray(x_train)
        n_features = x.shape[0]

        x_gaussian_copula = np.zeros_like(x, dtype=float)
        for j in range(n_features):
            vals = np.concatenate([[x[j]], x_train[:, j]])
            rank = rankdata(vals, method="average")[0]
            quantile = rank / (len(x_train) + 1)
            x_gaussian_copula[j] = norm.ppf(np.clip(quantile, 1e-10, 1 - 1e-10))
        return x_gaussian_copula

    def _inverse_transform(self, z_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original space."""
        x_original = np.zeros_like(z_samples)

        for i in range(z_samples.shape[1]):
            sorted_train = np.sort(self.data[:, i])
            quantile = norm.cdf(z_samples[:, i])
            ranks = quantile * (len(sorted_train) - 1)
            idx_low = np.floor(ranks).astype(int)
            idx_high = np.minimum(idx_low + 1, len(sorted_train) - 1)
            frac = ranks - idx_low
            x_original[:, i] = (1 - frac) * sorted_train[idx_low] + frac * sorted_train[idx_high]

        return x_original

    def impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Perform Gaussian Copula imputation for Shapley value calculations.

        Args:
            x: The data point to impute as an array of shape ``(n_features,)``.
            coalitions: Binary array of shape ``(n_coalitions, n_features)`` indicating which features are present (1) or missing (0)
                for each coalition.

        Returns:
            An array of shape ``(n_coalitions, n_features)`` containing the mean imputed values for each coalition in original feature space.
        """
        # Transform x_explain to Gaussian space for imputation
        x_explain = self._transform_x_explain(x.flatten(), self.data)
        imputed_data = self.sample_monte_carlo(x_explain, coalitions)

        # Invert transformation back to original feature space
        n_coalitions = imputed_data.shape[0]
        imputed_original = np.zeros_like(imputed_data)
        for i in range(n_coalitions):
            imputed_original[i] = self._inverse_transform(imputed_data[i])
        return np.mean(imputed_original, axis=1)
