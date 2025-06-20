"""Abstract base class for imputation approaches.

This module defines the base class for different imputation approaches used in SHAP value
calculations. It provides a common interface that all specific approaches must implement.
"""

from __future__ import annotations

from typing import Any

# TODO (milanagm): do we even need this script? I think we could already implement the specifics in the specific scripts


# TODO(Zaphood): Make this inherit from shapiq.imputer.base.Imputer
class GaussianImputerBase:
    """Abstract base class for imputation approaches.

    This class defines the interface that all specific imputation approaches must implement.
    """

    def __init__(self, internal: dict[str, Any]) -> None:
        """Initialize the approach with internal state.

        Parameters
        ----------
        internal : dict[str, Any]
            Dictionary containing the internal state and parameters for the approach.
        """
        self.internal = internal
        self.verbose = internal["parameters"]["verbose"]
        self.approach = internal["parameters"]["approach"]
