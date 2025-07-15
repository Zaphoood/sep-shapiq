"""Copula's approach for imputation in Shapley Value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm, rankdata

from .base import GaussianImputerBase

if TYPE_CHECKING:
    import numpy.typing as npt

X_CANNOT_BE_NONE_MSG = "Explanation point x cannot be None"


class GaussianCopulaImputer(GaussianImputerBase):
    """Implements the Gaussian Copula approach for feature imputation in Shapley Value calculations."""

    def __init__(
        self,
        model: object,
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        n_mc_samples: int = 1000,
        random_state: int | None = None,
    ) -> None:
        """Initializes the GaussianCopulaImputer.

        Args:
            model (object): The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data: The background data to use for the explainer as a 2-dimensional array with shape.
            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape or as a vector with shape. Defaults to None.
            n_mc_samples: Number of Monte Carlo samples for imputation. Defaults to 1000.
            random_state: The random state to use for sampling. Defaults to None.
                evaluation. Note that this can slow down the imputation process. Defaults to False.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
        )
        self._check_categorical_features()

        self.data_transformed = self.gaussian_transform(self.data)
        self._sorted_data = np.sort(self.data, axis=0)
        # Override: GaussianCopulaImputer uses transformed mean/covariance

        self._mean_per_feature = np.mean(
            self.data_transformed, axis=0
        )  # in theory mean should be (nearly) zero
        self._cov_mat = self._ensure_positive_definite(np.cov(self.data_transformed.T))

    def gaussian_transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
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

    def transform_x_explain(
        self, x_explain: npt.NDArray[np.floating], x_train: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Transform a single explanation point to Gaussian space using the training data's ECDF.

        Args:
            x_explain: The explanation point with shape (n_features,)
            x_train: The training data with shape (n_samples, n_features)

        Returns:
            Transformed explanation point in Gaussian space with shape (n_features,)
        """
        x = x_explain.flatten()
        x_train = np.asarray(x_train)
        n_features = x.shape[0]

        x_gaussian_copula = np.zeros_like(x, dtype=float)
        for j in range(n_features):
            vals = np.concatenate([[x[j]], x_train[:, j]])
            rank = rankdata(vals, method="average")[0]
            quantile = rank / (len(x_train) + 1)
            x_gaussian_copula[j] = norm.ppf(np.clip(quantile, 1e-10, 1 - 1e-10))
        return x_gaussian_copula

    def inverse_transform(self, z_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original space.

        Args:
            z_samples: Samples in Gaussian space (n_samples, n_features).

        Returns:
            Samples in original feature space (n_samples, n_features).
        """
        x_original = np.zeros_like(z_samples)

        for i in range(z_samples.shape[1]):
            sorted_train = self._sorted_data[:, i]
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
            x: The data point to impute as an array of shape (n_features,).
            coalitions: Boolean array of shape (n_coalitions, n_features) indicating which features
                are present or missing for each coalition.

        Returns:
            An array of shape (n_coalitions,) containing the mean model prediction for each coalition.
        """
        if x is None:
            raise ValueError(X_CANNOT_BE_NONE_MSG)

        # Transform explanation point to Gaussian space
        x_gaussian = self.transform_x_explain(x, self.data)

        # Generate Monte Carlo samples in Gaussian space
        gaussian_samples = self.sample_monte_carlo(x_gaussian, coalitions)

        # For each coalition, transform samples and get predictions
        n_coalitions = coalitions.shape[0]
        coalition_values = np.zeros(n_coalitions)

        for i in range(n_coalitions):
            # Transform samples back to original space
            original_samples = self.inverse_transform(gaussian_samples[i])
            # Get model predictions
            predictions = self.model(original_samples)
            # Take mean prediction for this coalition
            coalition_values[i] = np.mean(predictions)

        return coalition_values
