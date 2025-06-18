"""Main entry point - Data Imputation.

This module provides the main entry point for calculating SHAP values using different
data imputation approaches (Gaussian or Coppola). It handles the initialization of the
internal state and delegates the actual calculations to the appropriate approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
from .coppola_approach import CoppolaApproach
from .gaussian_approach import GaussianApproach


class InternalState(TypedDict):
    """This class stores the internal variable and parameter values when explain() is called."""

    parameters: dict[str, Any]
    data: dict[str, NDArray[np.float64]]
    iter_list: list[dict[str, Any]]
    timing_list: dict[str, Any]
    objects: dict[str, list[Any]]


class ExplainKwargs(TypedDict, total=False):
    """Optional keyword arguments for the `explain()` function.

    Parameters
    ----------
    verbose : list[str]
        List of strings for verbosity control.
    feature_names : list[str]
        List of names for each feature.
    n_MC_samples : int
        Number of Monte Carlo samples for estimating the Shapley values. Default is 1000.
    """

    verbose: list[str]
    feature_names: list[str]
    n_MC_samples: int


def explain(
    x_train: NDArray[np.float64],
    x_explain: NDArray[np.float64],
    approach: str = "gaussian",
    **kwargs: ExplainKwargs,
) -> InternalState:
    """Main function to explain predictions using either Gaussian or Coppola Imputation approach.

    Gaussian approach is used by default.

    Parameters
    ----------
    x_train : NDArray[np.float64]
        Training data used to fit the model.
    x_explain : NDArray[np.float64]
        Data to be explained.
    approach : str, optional
        The approach to use ('gaussian' or 'coppola'), by default 'gaussian'.
    **kwargs : ExplainKwargs
        Additional parameters for the specific approach:
        - verbose: List of strings for verbosity control
        - feature_names: List of feature names
        - n_MC_samples: Number of Monte Carlo samples

    Returns:
    -------
    InternalState
        Internal structure containing the explanation results.

    Raises:
    ------
    ValueError
        If an unknown approach is specified.
    """
    # Handle defaults
    # Use default feature names if none are provided
    feature_names = kwargs.get("feature_names", [f"X{i}" for i in range(x_train.shape[1])])
    # Number of Monte Carlo samples to use (default: 1000)
    n_MC_samples = kwargs.get("n_MC_samples", 1000)
    # List of strings for verbosity control (default: empty list)
    verbose = kwargs.get("verbose", [])

    # Initialize internal structure
    internal: InternalState = {
        "parameters": {
            "approach": approach,
            "feature_names": feature_names,
            "n_features": x_train.shape[1],
            "n_explain": x_explain.shape[0],
            "n_MC_samples": n_MC_samples,
            "verbose": verbose,
        },
        "data": {
            # Original training data
            "x_train": x_train,
            # Data to be explained
            "x_explain": x_explain,
        },
        # Placeholder for iterative process state (default: empty dict)
        "iter_list": [{}],
        # Placeholder for timing logs (default: empty dict)
        "timing_list": {},
        # Placeholder for feature specifications (default: empty list)
        "objects": {
            "feature_specs": [],
        },
    }

    # Select approach based on input
    if approach == "gaussian":
        approach_instance = GaussianApproach(internal)
    elif approach == "coppola":
        approach_instance = CoppolaApproach(internal)
    else:
        error_msg = (
            f"Unknown approach: {approach}. "
            "Please use either 'gaussian' or 'coppola' as approach parameter."
        )
        raise ValueError(error_msg)

    internal = approach_instance.setup_approach()
    return internal
