"""Gaussian Copula-based imputer for Shapley value estimation.

This module implements an imputation method using the Gaussian copula approach
for estimating the value function in Shapley-based explainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
from scipy.stats import norm, rankdata

from .base import GaussianImputerBase

if TYPE_CHECKING:
    import numpy.typing as npt


class GaussianCopulaImputer(GaussianImputerBase):
    """Copula-based Gaussian imputer using rank transformations.

    This imputer transforms data to Gaussian space using empirical CDF (rank-Gaussian transformation),
    performs imputation in the Gaussian space, and then transforms back to the original space.
    """

    @override
    def __init__(
        self,
        model: object,
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        n_mc_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """Initializes the Gaussian Copula imputer.

        Args:
            model: The model to explain as a callable function.
            data: Background data for the explainer (n_samples, n_features).
            x: Explanation point (1, n_features) or (n_features,). Defaults to None.
            n_mc_samples: Number of Monte Carlo samples. Defaults to 1000.
            random_state: Random state for reproducibility. Defaults to None.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
        )

        self._check_categorical_features()

        # Transform data to Gaussian space using empirical CDF (rank-Gaussian)
        self.data_transformed = self._rank_gaussian_transform(self.data)

        # Override: GaussianCopulaImputer uses transformed mean/covariance
        self._mean_per_feature = np.mean(
            self.data_transformed, axis=0
        )  # in theory mean should be (nearly) zero
        self._cov_mat = self._ensure_positive_definite(np.cov(self.data_transformed.T))

    def _rank_gaussian_transform(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform each feature to standard normal using empirical CDF (rank-Gaussian).

        Args:
            data: Input data array of shape (n_samples, n_features) to transform.

        Returns:
            Transformed data in Gaussian space with shape (n_samples, n_features).
        """
        data = np.asarray(data)
        n_samples, n_features = data.shape
        transformed_data = np.zeros_like(data, dtype=float)

        for i in range(n_features):
            # Use rankdata with average method to handle ties consistently
            ranks = rankdata(data[:, i], method="average")
            # Transform to uniform [0,1] using (rank - 0.5) / n to avoid extreme values
            uniform_values = (ranks - 0.5) / n_samples
            # Apply inverse normal CDF to get standard normal values
            transformed_data[:, i] = norm.ppf(np.clip(uniform_values, 1e-10, 1 - 1e-10))
        return transformed_data

    def _transform_point_to_gaussian(
        self, x_point: npt.NDArray[np.floating], x_train: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform explanation point to Gaussian space using training data's ECDF.

        Args:
            x_point: Explanation point to transform (n_features,).
            x_train: Training data used for ECDF (n_samples, n_features).

        Returns:
            Transformed point in Gaussian space (n_features,).
        """
        x_point = np.asarray(x_point).flatten()
        x_train = np.asarray(x_train)
        n_features = x_point.shape[0]
        n_samples = x_train.shape[0]

        x_gaussian_copula = np.zeros_like(x_point, dtype=float)

        for j in range(n_features):
            # Combine the point with reference data to get consistent ranking
            comb_value = np.concatenate([x_train[:, j], [x_point[j]]])
            rank = rankdata(comb_value, method="average")
            # Get the rank of our point (last element)
            uni_value = (rank[-1] - 0.5) / (n_samples + 1)
            x_gaussian_copula[j] = norm.ppf(np.clip(uni_value, 1e-10, 1 - 1e-10))
        return x_gaussian_copula

    def _transform_to_original(
        self, gaussian_samples: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original feature space.

        Args:
            gaussian_samples: Samples in Gaussian space (n_samples, n_features).

        Returns:
            Samples in original feature space (n_samples, n_features).
        """
        gaussian_samples = np.asarray(gaussian_samples)
        n_samples, n_features = gaussian_samples.shape
        x_original = np.zeros_like(gaussian_samples)

        for i in range(n_features):
            # Get uniform values from Gaussian samples
            uni_values = norm.cdf(gaussian_samples[:, i])

            # Use quantile function approach with sorted data
            n_ref = self._sorted_data.shape[0]

            # Convert uniform values to indices in sorted data
            indices = uni_values * (n_ref - 1)

            # Use numpy's interp for better interpolation
            x_original[:, i] = np.interp(indices, np.arange(n_ref), self._sorted_data[:, i])

        return x_original

    @override
    def impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Perform Gaussian Copula imputation for Shapley value calculations.

        Args:
            x: The data point to impute as an array of shape (n_features,).
            coalitions: Boolean array of shape (n_coalitions, n_features) indicating which features
                are present or missing for each coalition.

        Returns:
            An array of shape (n_coalitions,) containing the mean model prediction for each coalition.
        """
        NULL_POINT_MSG = "Explanation point x cannot be None"
        if x is None:
            raise ValueError(NULL_POINT_MSG)

        # Transform explanation point to Gaussian space
        x_gaussian_copula = self._transform_point_to_gaussian(x.flatten(), self.data)
        # Generate Monte Carlo samples in Gaussian space
        gaussian_samples = self.sample_monte_carlo(x_gaussian_copula, coalitions)

        # For each coalition, transform samples and get predictions
        n_coalitions = coalitions.shape[0]
        coalition_values = np.zeros(n_coalitions)

        for i in range(n_coalitions):
            # Transform samples back to original space
            original_samples = self._transform_to_original(gaussian_samples[i])
            # Get model predictions
            predictions = self.model(original_samples)
            # Take mean prediction for this coalition
            coalition_values[i] = np.mean(predictions)

        return coalition_values
