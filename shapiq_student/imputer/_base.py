"""Abstract base class for imputation approaches.

This module defines the base class for different imputation approaches used in SHAP value
calculations. It provides a common interface that all specific approaches must implement.
"""

# TODO(Zaphoood): Make sure all docstrings follow Google's style

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from .exceptions import FeatureNamesLengthError

# TODO (milanagm): do we even need this script? I think we could already implement the specifics in the specific scripts


# TODO(Zaphood): Make this inherit from shapiq.imputer.base.Imputer
class GaussianImputerBase:
    """Abstract base class for imputation approaches.

    This class defines the interface that all specific imputation approaches must implement.
    """

    def __init__(
        self,
        data: npt.NDArray[np.floating],
        n_mc_samples: int = 1000,
        feature_names: list[str] | None = None,
        verbose: list[str] | None = None,
    ) -> None:
        """Initializes GaussianImputerBase.

        Args:
            data: Training data used to fit the model.
            verbose: List of strings for verbosity control, by default None
            feature_names: Optional list of feature names. If not provided, feature names are generated automatically
            n_mc_samples: Number of Monte Carlo samples, by default 1000
        """
        # We don't need to pass approach, since it is defined by instantiating a
        #   subclass (GaussianImputer or GaussianCopulaImputer)  # noqa: ERA001
        # self.approach = internal["parameters"]["approach"]  # noqa: ERA001

        # TODO(Zaphoood): Use python's `logging` library instead
        self.verbose = verbose

        self.data = data
        self.n_features = data.shape[1]
        self.n_mc_samples = n_mc_samples

        # TODO(Zaphoood): maybe we don't need feature names at all?
        if feature_names is None:
            self.feature_names = [
                f"X{i}" for i in range(self.n_features)
            ]  # If, after DataFrame checks, we still have no names, build a list ["X0","X1",…]
        elif len(feature_names) != self.n_features:
            # If passed count of feature names doesn't match number of columns
            raise FeatureNamesLengthError(len(feature_names), self.n_features)
        else:
            self.feature_names = feature_names

        # TODO(Zaphoood): x_explain is passed later in some function (?)
        # self.x_explain = x_explain_arr  # noqa: ERA001
        # self.n_explain = x_explain_arr.shape[0]  # noqa: ERA001

        # TODO(Zaphood): maybe iter_list and timing_list
