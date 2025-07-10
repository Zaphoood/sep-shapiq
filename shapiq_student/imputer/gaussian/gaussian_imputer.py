"""Gaussian approach for imputation in Shapley Value calculations.

This module implements a Gaussian-based approach for generating samples in Shapley Value
calculations. It uses multivariate normal distribution for sampling and handles
continuous features only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from shapiq import Game

from .base import GaussianImputerBase


class GaussianImputer(GaussianImputerBase):
    """Implementation of Gaussian-based approach for Shapley Value calculations.

    This approach uses multivariate normal distribution for generating samples.
    It only supports continuous features.
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
        """Initializes the GaussianImputer.

        Args:
            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data: The background data to use for the explainer as a 2-dimensional array with shape
                (n_samples, n_features).
            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape (1, n_features) or as a vector with shape (n_features,). Defaults to None.
            n_mc_samples: Number of Monte Carlo samples for imputation. Defaults to 1000.
            random_state: The random state to use for sampling. Defaults to None.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            n_mc_samples=n_mc_samples,
            random_state=random_state,
        )
        self._check_categorical_features()

    def get_imputed_result_data(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Impute missing values for given coalitions using Gaussian MC sampling.

        This method performs the imputation without calling the prediction function.

        Args:
            coalitions: Binary array indicating which features are present (1) or missing (0)
                for each coalition. Shape: (n_coalitions, n_features).

        Returns:
            Imputed data points for each coalition, averaged over Monte Carlo samples.
            Shape: (n_coalitions, n_features).

        Raises:
            RuntimeError: If the explanation point x is not set.
        """
        if self.x is None:
            msg = f"Must call {self.__class__.__name__}.fit() first before imputing"
            raise RuntimeError(msg)

        return np.mean(self.impute(self.x, coalitions), axis=1)

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Impute missing values and return model predictions for given coalitions.

        This method performs imputation and then calls the model's prediction function
        on the imputed data. This is the main interface expected by Shapley Value explainers.

        Args:
            coalitions: Binary array indicating which features are present (1) or missing (0)
                for each coalition. Shape: (n_coalitions, n_features).

        Returns:
            Model predictions for each imputed data point.
                Shape: (n_explain * n_coalitions * n_mc_samples,).

        Raises:
            RuntimeError: If the explanation point x is not set.
        """
        return self.predict(self.get_imputed_result_data(coalitions))
