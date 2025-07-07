"""Copula's approach for imputation in SHAP value calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, rankdata

from .base import GaussianImputerBase

if TYPE_CHECKING:
    import numpy.typing as npt


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
        verbose: bool = False,
    ) -> None:
        """Initializes the GaussianCopulaImputer.

        Args:
            model (object): The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data (npt.NDArray[np.floating]): The background data to use for the explainer as a 2-dimensional array with shape
                (n_samples, n_features).
            x (npt.NDArray[np.floating] | None, optional): The explanation point to use the imputer on either as a 2-dimensional array with
                shape (1, n_features) or as a vector with shape (n_features,). Defaults to None.
            n_mc_samples (int, optional): Number of Monte Carlo samples for imputation. Defaults to 1000.
            random_state (int | None, optional): The random state to use for sampling. Defaults to None.
            verbose (bool, optional): A flag to enable verbose imputation, which will print a progress bar for model
                evaluation. Note that this can slow down the imputation process. Defaults to False.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
            verbose=verbose,
        )
        self._check_categorical_features()
        self._data_gaussian = self._gaussian_transform(self.data)
        self._mean_per_feature_gaussian = np.mean(self._data_gaussian, axis=0)
        self._cov_mat_gaussian = self._ensure_positive_definite(np.cov(self._data_gaussian.T))

    def _initialize_copula_parameters(self) -> None:
        """Initialize Gaussian-transformed data and statistics."""
        self._data_gaussian = self._gaussian_transform(self.data)
        self._mean_gaussian = np.mean(self._data_gaussian, axis=0)
        self._cov_gaussian = self._ensure_positive_definite(np.cov(self._data_gaussian.T))

    def _gaussian_transform(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform data to standard normal distribution using empirical CDF."""
        ranks = rankdata(x, method="average")
        u = ranks / (len(ranks) + 1)
        return norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))

    def _transform_explain_data(self, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Transform explanation data to Gaussian space using training data's ECDF."""
        x_gaussian = np.zeros_like(x)
        for i in range(x.shape[1]):
            combined = np.vstack([x[:, [i]], self.data[:, [i]]])
            ranks = rankdata(combined, axis=0, method="average")
            explain_ranks = ranks[: x.shape[0]]

            adjusted_ranks = explain_ranks - rankdata(explain_ranks, axis=0, method="average") + 0.5
            u = adjusted_ranks / (len(combined) - x.shape[0] + 1)
            x_gaussian[:, i] = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        return x_gaussian

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

    def copula_imputation(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Perform Gaussian Copula imputation for SHAP value calculations.

        Returns:
        -------
        np.ndarray
            A 3D array of shape (n_MC_samples, n_explain * n_coalitions, n_features)
            containing all imputed samples in original feature space.
        """
        if self.x is None:
            msg = "Explanation point x must be set first"
            raise RuntimeError(msg)

        x = np.atleast_2d(self.x)
        x_gaussian = self._transform_explain_data(x)
        n_coalitions, n_features = coalitions.shape
        rng = default_rng(self.random_state)

        result_cube = np.zeros((n_coalitions, self.n_mc_samples, n_features))

        for S_ind in range(n_coalitions):
            for _idx_now in range(coalitions.shape[0]):
                S_known = np.where(coalitions)[0]
                S_unknown = np.where(~coalitions)[0]

                if len(S_known) == 0:
                    # No conditioning - sample from marginal
                    Z = rng.standard_normal((self.n_mc_samples, n_features))
                    samples_gaussian = Z @ np.linalg.cholesky(self._cov_gaussian).T
                elif len(S_unknown) == 0:
                    # All features known - just repeat the explicand
                    samples_gaussian = np.tile(x_gaussian[0], (self.n_mc_samples, 1))
                else:
                    # Conditional sampling
                    cov_SS = self._cov_gaussian[np.ix_(S_known, S_known)]
                    cov_SuS = self._cov_gaussian[np.ix_(S_unknown, S_known)]
                    cov_SuSu = self._cov_gaussian[np.ix_(S_unknown, S_unknown)]

                    cov_SS_inv = np.linalg.inv(cov_SS)
                    cond_mean = self._mean_gaussian[S_unknown] + cov_SuS @ cov_SS_inv @ (
                        x_gaussian[0, S_known] - self._mean_gaussian[S_known]
                    )
                    cond_cov = cov_SuSu - cov_SuS @ cov_SS_inv @ cov_SuS.T
                    cond_cov = 0.5 * (cond_cov + cond_cov.T)  # Ensure symmetry

                    Z = rng.standard_normal((self.n_mc_samples, len(S_unknown)))
                    samples_unknown = Z @ np.linalg.cholesky(cond_cov).T + cond_mean

                    samples_gaussian = np.tile(x_gaussian[0], (self.n_mc_samples, 1))
                    samples_gaussian[:, S_unknown] = samples_unknown

                result_cube[S_ind] = self._inverse_transform(samples_gaussian)

        return np.mean(result_cube, axis=1)

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Compute model predictions for imputed coalitions."""
        return self.predict(self.copula_imputation(coalitions))
