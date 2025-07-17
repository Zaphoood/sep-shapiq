"""Gaussian approach for imputation in Shapley Value calculations.

This module implements a Gaussian-based approach for generating samples in Shapley Value
calculations. It uses multivariate normal distribution for sampling and handles
continuous features only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from .base import GaussianImputerBase


class GaussianImputer(GaussianImputerBase):
    """Implements a Gaussian-based approach for imputation.

    This approach assumes that the features of the background data form a multivariate Gaussian distribution.
    Missing values are imputed by first calculating the conditional distribution of missing features given the present
    features and the using Monte Carlo sampling to generate values.

    Note that only continuous features are supported, meaning that this imputer can't be used for datasets containing
    categorical or binary features.
    """

    def _impute(
        self, x: npt.NDArray[np.floating], coalitions: npt.NDArray[np.bool]
    ) -> npt.NDArray[np.floating]:
        mc_samples = self.sample_monte_carlo(x, coalitions)
        return cast("npt.NDArray[np.floating]", np.mean(mc_samples, axis=1))
