"""Abstract base class for imputation approaches.

This module defines the base class for different imputation approaches used in SHAP value
calculations. It provides a common interface that all specific approaches must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

import numpy as np

# TODO (milanagm): do we even need this script? I think we could already implement the specifics in the specific scripts


class Approach(ABC):
    """Abstract base class for imputation approaches.

    This class defines the interface that all specific imputation approaches must implement.
    It handles common setup logic and provides abstract methods for approach-specific
    implementations.
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

    def setup_approach(self) -> dict[str, Any]:
        """Main setup method that handles common logic.

        Returns:
        -------
        dict[str, Any]
            Updated internal state dictionary.
        """
        # Logging if verbose
        if "progress" in self.verbose:
            logging.info("Setting up %s approach", self.approach)

        # Call specific setup
        self._setup_specific()

        # Update timing
        self.internal["timing_list"]["setup_approach"] = np.datetime64("now")

        return self.internal

    @abstractmethod
    def _setup_specific(self) -> None:
        """Specific setup implementation for each approach.

        This method must be implemented by concrete approach classes to handle
        approach-specific setup logic.
        """
