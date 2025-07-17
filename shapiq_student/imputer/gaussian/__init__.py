"""Implementations of Gaussian-based imputers.

This package provides two imputer implementations:

- ``GaussianImputer``: Uses Monte Carlo sampling from normal distributions of features.
- ``GaussianCopulaImputer``: Uses Gaussian copulas to model arbitrary data as normal distributions.
"""

from .copula_imputer import GaussianCopulaImputer
from .gaussian_imputer import GaussianImputer

__all__ = ["GaussianImputer", "GaussianCopulaImputer"]
