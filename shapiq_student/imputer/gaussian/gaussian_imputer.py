"""Gaussian approach for imputation in Shapley Value calculations.

This module implements a Gaussian-based approach for generating samples in Shapley Value
calculations. It uses multivariate normal distribution for sampling and handles
continuous features only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

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

    def impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        """Impute missing values for given coalitions using Gaussian MC sampling.

        This method performs the imputation without calling the prediction function.

        Args:
            x: The data point to impute as an array of shape ``(n_features,)``.
            coalitions: Boolean array of shape ``(n_coalitions, n_features)`` indicating which features are present or missing for each coalition.

        Returns:
            An array of shape ``(n_coalitions, n_features)`` containing the imputed data points for each coalition, averaged over Monte Carlo samples.
        """
        return cast(
            "npt.NDArray[np.floating]", np.mean(self.sample_monte_carlo(x, coalitions), axis=1)
        )
