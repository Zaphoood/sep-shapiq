"""Gaussian approach for imputation in SHAP value calculations.

This module implements a Gaussian-based approach for generating samples in SHAP value
calculations. It uses multivariate normal distribution for sampling and handles
continuous features only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.random import default_rng

if TYPE_CHECKING:
    import numpy.typing as npt

from .base import GaussianImputerBase


class GaussianImputer(GaussianImputerBase):
    """Implementation of Gaussian-based approach for SHAP value calculations.

    This approach uses multivariate normal distribution for generating samples.
    It only supports continuous features and raises an error if categorical features
    are detected.
    """

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
        """Initializes the GaussianImputer.

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

    def get_imputed_result_data_gaussian(
        self, coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Impute missing values for given coalitions using Gaussian MC sampling.

        This method performs the imputation but returns the imputed data cube without
        calling the prediction function. Useful for testing and debugging.

        Args:
            coalitions (npt.NDArray[np.bool]): Binary array indicating which features are present (1) or missing (0)
                for each coalition. Shape: (n_coalitions, n_features).

        Returns:
            npt.NDArray[np.floating]: Imputed data points for each explanation point and coalition.
                Shape: (n_explain, n_coalitions, n_mc_samples, n_features).

        Raises:
            RuntimeError: If the explanation point x is not set.
        """
        if self.x is None:
            msg = f"Must call {self.__class__.__name__}.fit() first before imputing"
            raise RuntimeError(msg)

        x_explain = self.x.flatten()
        n_coalitions, n_features = coalitions.shape
        n_mc_samples = self.n_mc_samples
        mean_per_feature = self.mean_per_feature
        cov_mat = self.cov_mat
        rng = default_rng(self.random_state)

        result_cube = np.zeros((n_coalitions, n_mc_samples, n_features))

        for S_ind, coalition in enumerate(coalitions):
            S_idx_known = np.where(coalition)[0]
            S_idx_unknown = np.where(~coalition)[0]

            if len(S_idx_known) == 0:
                # No conditioning on known features, therefore sample from original data distribution
                Z = rng.standard_normal((n_mc_samples, len(S_idx_unknown)))
                samples = Z @ np.linalg.cholesky(cov_mat).T + mean_per_feature
            if len(S_idx_unknown) == 0:
                # all features known, therefore we just return explanation point
                samples = np.tile(x_explain, (n_mc_samples, 1))
                # TODO (milanagm): aus dem loop aussteigen und x_explain direkt unten in samples eintragen
            else:
                x_S_star = x_explain[S_idx_known]

                mu_S_known = mean_per_feature[S_idx_known]
                mu_S_unknown = mean_per_feature[S_idx_unknown]

                cov_S_known_known = cov_mat[np.ix_(S_idx_known, S_idx_known)]
                cov_S_known_unknown = cov_mat[np.ix_(S_idx_known, S_idx_unknown)]
                cov_S_unknown_known = cov_mat[np.ix_(S_idx_unknown, S_idx_known)]
                cov_S_unknown_unknown = cov_mat[np.ix_(S_idx_unknown, S_idx_unknown)]

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
                Z = rng.standard_normal((n_mc_samples, len(S_idx_unknown)))
                samples_unknown = Z @ np.linalg.cholesky(cond_cov).T + cond_mean

                samples = np.tile(x_explain, (n_mc_samples, 1))
                samples[:, S_idx_unknown] = samples_unknown

            result_cube[S_ind] = samples

        return np.mean(result_cube, axis=1)

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Impute missing values and return model predictions for given coalitions.

        This method performs imputation and then calls the model's prediction function
        on the imputed data. This is the main interface expected by SHAP explainers.

        Args:
            coalitions (npt.NDArray[np.bool]): Binary array indicating which features are present (1) or missing (0)
                for each coalition. Shape: (n_coalitions, n_features).

        Returns:
            npt.NDArray[np.floating]: Model predictions for each imputed data point.
                Shape: (n_explain * n_coalitions * n_mc_samples,).

        Raises:
            RuntimeError: If the explanation point x is not set.
        """
        return self.predict(self.get_imputed_result_data_gaussian(coalitions))
