"""Imputation approaches for SHAP value calculations.

This module provides different approaches for handling missing values and generating
samples for SHAP value calculations. It includes two main approaches:

- GaussianApproach: Uses Gaussian distribution for generating samples
- CoppolaApproach: Uses Coppola's binning strategy for generating samples

The main entry point is the `explain()` function, which provides a unified interface
for both approaches.

Classes
-------
Approach : ABC
    Abstract base class defining the interface for imputation approaches
GaussianApproach : Approach
    Implementation using Gaussian distribution for sampling
CoppolaApproach : Approach
    Implementation using Coppola's binning strategy

Functions
---------
explain : function
    Main function to explain predictions using either Gaussian or Coppola approach
"""

from .approach import Approach
from .coppola_approach import CoppolaApproach
from .gaussian_approach import GaussianApproach
from .main import explain

__all__ = ["Approach", "GaussianApproach", "CoppolaApproach", "explain"]
