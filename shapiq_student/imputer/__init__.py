"""Imputation approaches for SHAP value calculations.

This module provides different approaches for handling missing values and generating
samples for SHAP value calculations. It includes two main approaches:

- GaussianApproach: Uses Gaussian distribution for generating samples
- CopulaApproach: Uses Copula's binning strategy for generating samples

The main entry point is the `explain()` function, which provides a unified interface
for both approaches.

Classes
-------
Approach : ABC
    Abstract base class defining the interface for imputation approaches
GaussianApproach : Approach
    Implementation using Gaussian distribution for sampling
CopulaApproach : Approach
    Implementation using Copula's binning strategy

Functions
---------
explain : function
    Main function to explain predictions using either Gaussian or Copula approach
"""

from .approach import Approach
from .Copula_approach import CopulaApproach
from .gaussian_approach import GaussianApproach
from .main import explain

__all__ = ["Approach", "GaussianApproach", "CopulaApproach", "explain"]
