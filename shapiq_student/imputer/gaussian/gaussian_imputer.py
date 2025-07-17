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
    """Implements a Gaussian-based approach for imputation.

    This approach assumes that the features of the background data form a multivariate Gaussian distribution.
    Missing values are imputed by first calculating the conditional distribution of missing features given the present
    features and the using Monte Carlo sampling to generate values.

    Note that only continuous features are supported, meaning that this imputer can't be used for datasets containing
    categorical or binary features.
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

    def _impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        return cast(
            "npt.NDArray[np.floating]", np.mean(self.sample_monte_carlo(x, coalitions), axis=1)
        )
