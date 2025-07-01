"""Imputation approaches for SHAP value calculations.

This module provides different approaches for handling missing values and generating
samples for SHAP value calculations. It includes two main approaches:

- GaussianImputer: Uses Gaussian distribution for generating samples
- GaussianCopulaImputer: Uses Copula's binning strategy for generating samples

Classes
-------
GaussianImputerBase : Imputer
    Abstract base class for Gaussian-based imputation approaches
GaussianImputer : GaussianImputerBase
    Implementation using Gaussian distribution for sampling
GaussianCopulaImputer : GaussianImputerBase
    Implementation using Copula's binning strategy
"""

from .gaussian_copula_imputer import GaussianCopulaImputer
from .gaussian_imputer import GaussianImputer

__all__ = ["GaussianImputer", "GaussianCopulaImputer"]
