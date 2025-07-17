"""Copula's approach for imputation in Shapley Value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

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

        self.data_transformed = self.transform_to_gaussian(self.data)
        # The mean should be zero in theory but may differ in practice, therefore we still need to compute it
        self._mean_per_feature = np.mean(self.data_transformed, axis=0)
        self._cov_mat = self._ensure_positive_definite(np.cov(self.data_transformed.T))
        # Sorted data is required for the transformation back from Gaussian space to the original feature space
        self._data_sorted = np.sort(self.data, axis=0)

    def transform_to_gaussian(
        self, background_data: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform each feature to a standard normal distribution using empirical CDF (rank-Gaussian).

        For each feature (column), this method applies a transformation so that the values follow a standard normal
        distribution (mean 0, std 1), while preserving the rank order of the original data. This is also known as a
        rank-Gaussian or empirical CDF transformation.

        Args:
            background_data: Input data to transform as an array of shape ``(n_samples, n_features)``.

        Returns:
            Transformed data in Gaussian space as an array of shape ``(n_samples, n_features)``.
        """
        transformed = np.zeros_like(background_data, dtype=float)

        for col in range(background_data.shape[1]):
            empirical_cdf = self._empirical_cdf(background_data[:, col])
            # TODO(Zaphoood): clipping shouldn't be necessary, since quantiles are already in range (0, 1)
            transformed[:, col] = norm.ppf(np.clip(empirical_cdf, 1e-10, 1 - 1e-10))

        return transformed

    def _empirical_cdf(self, samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Computes the empirical cumulative distribution function for the given samples.

        Note that we define the empirical distribution in such a way that `f(x_min) == 1/(n+1)`
        and `f(x_max) == n/(n+1)`, where `x_min` and `x_max` are the minimal and maximal obeserved sample respectively.
        """
        ranks = rankdata(samples, method="average")
        # Map ranks linearly to the range [1/(n+1), n/(n+1)]
        edf = ranks / (len(ranks) + 1)

        return edf

    def transform_point_to_gaussian(
        self,
        background_data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Transform a single explanation point to Gaussian space using the training data's ECDF.

        Args:
            background_data: Training data used to compute ECDF, with shape ``(n_samples, n_features)``.
            x: Explanation point to transform, with shape ``(n_features,)``.

        Returns:
            Transformed point in Gaussian space, with shape ``(n_features,)``.
        """
        n_features = background_data.shape[1]

        if x.shape[0] != background_data.shape[1]:
            msg = (
                f"Background data has {n_features} features but point to transform has {x.shape[0]}"
            )
            raise ValueError(msg)

        x_transformed = np.zeros_like(x, dtype=float)

        for col in range(n_features):
            rank = cast("np.integer", np.sum(background_data[:, col] <= x[col]))
            # Need to clip the rank, since it may be equal to 0 (or n_train) if
            # the value is smaller (or larger) than the smallest (largest)
            # sample in the background data
            rank = np.clip(rank, a_min=1, a_max=background_data.shape[0])

            quantile = rank / (background_data.shape[0] + 1)
            # TODO(Zaphoood): clipping should be unnecessary here
            x_transformed[col] = norm.ppf(np.clip(quantile, 1e-10, 1 - 1e-10))

        return x_transformed

    def transform_from_gaussian(
        self, data_gaussian: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original feature space.

        The method np.interp is a linear interpolation method, which means that after transforming
        back from Gaussian space, it estimates the closest values based on the sorted data.
        This doesn't necessarily match the exact values of the original data.

        Args:
            data_gaussian: Samples in Gaussian space ``(n_samples, n_features)``.

        Returns:
            Samples in original feature space ``(n_samples, n_features)``.
        """
        n_features = data_gaussian.shape[1]
        n_samples = self.data.shape[0]

        x_original = np.zeros_like(data_gaussian)
        for col in range(n_features):
            quantiles = norm.cdf(data_gaussian[:, col])
            ranks = quantiles * (n_samples + 1)
            # The back-transformed ranks are not necessarily integers, so we interpolate linearly between the closest original datapoints
            x_original[:, col] = np.interp(
                ranks, np.arange(1, n_samples + 1), self._data_sorted[:, col]
            )

        return x_original

    def _impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        x_transformed = self.transform_point_to_gaussian(self.data, x.flatten())

        gaussian_samples = self.sample_monte_carlo(x_transformed, coalitions)

        samples_backtransformed = np.zeros_like(gaussian_samples)
        for coal_idx in range(coalitions.shape[0]):
            samples_backtransformed[coal_idx] = self.transform_from_gaussian(
                gaussian_samples[coal_idx]
            )
        imputed = cast("npt.NDArray[np.floating]", np.mean(samples_backtransformed, axis=1))

        return imputed
