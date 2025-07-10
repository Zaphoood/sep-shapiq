"""Copula's approach for imputation in SHAP value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, rankdata

if TYPE_CHECKING:
    import numpy.typing as npt

from .base import GaussianImputerBase


class GaussianCopulaImputer(GaussianImputerBase):
    """Implements the Gaussian Copula approach for feature imputation in SHAP value calculations."""

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
        self._initialize_copula_parameters()

    def _initialize_copula_parameters(self) -> None:
        """Transform training data to Gaussian space and compute mean and covariance.

        This method applies a Gaussian (normal) transformation to each feature in the training data
        and then calculates the mean vector and covariance matrix in the transformed (Gaussian) data set.
        """
        self._data_transformed = self._gaussian_transform(self.data)
        self._mean = np.mean(
            self._data_transformed, axis=0
        )  # in theory mean should be (nearly) zero
        self._cov = self._ensure_positive_definite(np.cov(self._data_transformed.T))

    def _gaussian_transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform each feature to standard normal using empirical CDF (rank-Gaussian).

        For each feature (column), this method applies a transformation so that the values follow a standard normal
        distribution (mean 0, std 1), while preserving the rank order of the original data. This is also known as a
        rank-Gaussian or empirical CDF transformation.
        """
        x_trans = np.zeros_like(x, dtype=float)
        for i in range(x.shape[1]):
            ranks = rankdata(x[:, i], method="average")
            u = ranks / (len(ranks) + 1)
            x_trans[:, i] = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        return x_trans

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
            u = rank / (len(x_train) + 1)
            x_gaussian_copula[j] = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        return x_gaussian_copula

    def _inverse_transform(self, z_samples: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform Gaussian samples back to original space."""
        x_original = np.zeros_like(z_samples)

        for i in range(z_samples.shape[1]):
            sorted_train = np.sort(self.data[:, i])
            u = norm.cdf(z_samples[:, i])
            ranks = u * (len(sorted_train) - 1)
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
            An array of shape (n_MC_samples, n_explain * n_coalitions, n_features)
            containing all imputed samples in original feature space.
        """
        if self.x is None:
            msg = f"Must call {self.__class__.__name__}.fit() first before imputing"
            raise RuntimeError(msg)

        x_gaussian_copula = self._transform_x_explain(self.x.flatten(), self.data)
        n_coalitions, n_features = coalitions.shape
        n_mc_samples = self.n_mc_samples
        mean_gaussian_copula = self._mean
        cov_gaussian_copula = self._cov
        rng = default_rng(self.random_state)

        result_cube = np.zeros((n_coalitions, n_mc_samples, n_features))

        for S_ind, coalition in enumerate(coalitions):
            S_idx_known = np.where(coalition)[0]
            S_idx_unknown = np.where(~coalition)[0]

            if len(S_idx_known) == 0:
                # No conditioning - sample from marginal
                Z = rng.standard_normal((n_mc_samples, n_features))
                samples_gaussian = Z @ np.linalg.cholesky(cov_gaussian_copula).T
            elif len(S_idx_unknown) == 0:
                # All features known - just repeat the explicand
                samples_gaussian = np.tile(x_gaussian_copula[0], (n_mc_samples, 1))
            else:
                x_S_star = x_gaussian_copula[S_idx_known]

                mu_S_known = mean_gaussian_copula[S_idx_known]
                mu_S_unknown = mean_gaussian_copula[S_idx_unknown]

                cov_S_known_known = cov_gaussian_copula[np.ix_(S_idx_known, S_idx_known)]
                cov_S_known_unknown = cov_gaussian_copula[np.ix_(S_idx_known, S_idx_unknown)]
                cov_S_unknown_known = cov_gaussian_copula[np.ix_(S_idx_unknown, S_idx_known)]
                cov_S_unknown_unknown = cov_gaussian_copula[np.ix_(S_idx_unknown, S_idx_unknown)]

                cov_S_known_known_inv = np.linalg.inv(cov_S_known_known)

                cond_mean = mu_S_unknown + (cov_S_unknown_known @ cov_S_known_known_inv) @ (
                    x_S_star - mu_S_known
                )
                cond_cov = (
                    cov_S_unknown_unknown
                    - (cov_S_unknown_known @ cov_S_known_known_inv) @ cov_S_known_unknown
                )
                cond_cov = 0.5 * (cond_cov + cond_cov.T)  # Ensure symmetry

                Z = rng.standard_normal((n_mc_samples, len(S_idx_unknown)))
                samples_unknown = Z @ np.linalg.cholesky(cond_cov).T + cond_mean

                samples_gaussian = np.tile(x_gaussian_copula[0], (n_mc_samples, 1))
                samples_gaussian[:, S_idx_unknown] = samples_unknown

            result_cube[S_ind] = self._inverse_transform(samples_gaussian)

        return np.mean(result_cube, axis=1)

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Compute model predictions for imputed coalitions."""
        return self.predict(self.get_imputed_result_data(coalitions))
