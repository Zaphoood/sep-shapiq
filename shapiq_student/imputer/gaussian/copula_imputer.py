"""Copula's approach for imputation in Shapley Value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm, rankdata

if TYPE_CHECKING:
    import numpy.typing as npt

from .base import GaussianImputerBase


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
            x_explain: The explanation point with shape (n_features,)
            x_train: The training data with shape (n_samples, n_features)

        Returns:
            Transformed explanation point in Gaussian space with shape (n_features,)
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

    def get_imputed_result_data(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Perform Gaussian Copula imputation for Shapley value calculations.

        Returns:
        -------
        np.ndarray
        An array of shape (n_coalitions, n_features) containing the mean imputed values for each coalition in original feature space.
        """
        if self.x is None:
            msg = f"Must call {self.__class__.__name__}.fit() first before imputing"
            raise RuntimeError(msg)

        # Transforming x_explain to Gaussian space for imputation
        x_explain = self._transform_x_explain(self.x.flatten(), self.data)
        imputed_data = self.impute(x_explain, coalitions)

        # Inverting transformation back to original feature space
        n_coalitions, n_mc_samples, n_features = imputed_data.shape
        imputed_original = np.zeros_like(imputed_data)
        for i in range(n_coalitions):
            imputed_original[i] = self._inverse_transform(imputed_data[i])
        return np.mean(imputed_original, axis=1)

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Compute model predictions for imputed coalitions."""
        return self.predict(self.get_imputed_result_data(coalitions))
