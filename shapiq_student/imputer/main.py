"""Main entry point - Data Imputation.

This module provides the main entry point for calculating SHAP values using different
data imputation approaches (Gaussian or Copula). It handles the initialization of the
internal state and delegates the actual calculations to the appropriate approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

from .Copula_approach import CopulaApproach
from .gaussian_approach import GaussianApproach


class FeatureNamesLengthError(ValueError):
    """Exception raised when feature names length doesn't match number of features."""

    def __init__(self, feature_names_length: int, n_features: int) -> None:
        """Initialize the error with the mismatched lengths.

        Parameters
        ----------
        feature_names_length : int
            Length of the provided feature names list
        n_features : int
            Number of features in the data
        """
        self.feature_names_length = feature_names_length
        self.n_features = n_features
        message = f"feature_names length {feature_names_length} != number of features {n_features}"
        super().__init__(message)


class InternalState(TypedDict):
    """This class stores the internal variable and parameter values when explain() is called."""

    parameters: dict[str, Any]
    data: dict[str, NDArray[np.float64]]
    iter_list: list[
        dict[str, Any]
    ]  # TODO (milanagm): if we decide to take this variable, also delete this line
    timing_list: dict[
        str, Any
    ]  # TODO (milanagm): if we decide to take this variable, also delete this line
    objects: dict[
        str, list[Any]
    ]  # TODO (milanagm): if we decide to take out the feature_spec variable, also delete this line


class ExplainKwargs(TypedDict, total=False):
    """Type definition for the explain function's keyword arguments.

    All parameters (verbose, feature_names and n_MC_samples)can be optionally passed within explain() method
    """

    verbose: list[str]  # if we want specific logging
    feature_names: list[str]
    n_MC_samples: int  # TODO (milanagm): does it make sense to make this passable in explain()?


def explain(  # TODO (milanagm): we need to add tests - how should that look like?
    x_train: pd.DataFrame | NDArray[np.float64],  # accepting both pandas and numpy
    x_explain: pd.DataFrame | NDArray[np.float64],
    approach: str = "gaussian",
    **kwargs: ExplainKwargs,  # this enables the variables in ExplainKwargs to be passed (optionally)
) -> InternalState:
    """Main function to explain predictions using either Gaussian or Copula Imputation approach.

    This function handles both pandas DataFrame and numpy array inputs. For DataFrames, it automatically
    extracts feature names from column names if not provided. For numpy arrays, it generates default
    feature names (X0, X1, etc.) if none are provided.

    Parameters
    ----------
    x_train : Union[pd.DataFrame, NDArray[np.float64]]
        Training data used to fit the model. Can be either a pandas DataFrame or numpy array.
    x_explain : Union[pd.DataFrame, NDArray[np.float64]]
        Data to be explained. Can be either a pandas DataFrame or numpy array.
    approach : str, optional
        The approach to use ('gaussian' or 'Copula'), by default 'gaussian'.
    **kwargs : ExplainKwargs
        Additional parameters for the specific approach:
        - verbose: List of strings for verbosity control, by default []
        - feature_names: List of feature names, by default None (extracted from DataFrame columns
          or generated as X0, X1, etc. for numpy arrays)
        - n_MC_samples: Number of Monte Carlo samples, by default 1000

    Returns:
    -------
    InternalState
        Internal structure containing the explanation results, including:
        - parameters: Dictionary of parameters used
        - data: Dictionary containing x_train and x_explain as numpy arrays
        - iter_list: List for iteration tracking
        - timing_list: Dictionary for timing information

    Raises:
    ------
    ValueError
        If an unknown approach is specified.
    FeatureNamesLengthError
        If the length of provided feature_names doesn't match the number of features.
    """
    # TODO (milanagm): add tests for set_up?

    # 1) Handle DataFrame inputs and extract columns if provided
    feature_names: Sequence[str] | None = kwargs.get("feature_names")
    if isinstance(x_train, pd.DataFrame):
        if feature_names is None:
            feature_names = list(x_train.columns)  # extracts column names if not in kwargs
        x_train_arr = x_train.to_numpy()  # convert to numpy
    else:
        x_train_arr = x_train

    if isinstance(x_explain, pd.DataFrame):
        if feature_names is None:
            feature_names = list(x_explain.columns)
        x_explain_arr = x_explain.to_numpy()
    else:
        x_explain_arr = x_explain

    # 2) Build default names if still missing and validate
    n_features = x_train_arr.shape[1]
    if feature_names is None:
        feature_names = [
            f"X{i}" for i in range(n_features)
        ]  # If, after DataFrame checks, we still have no names, build a list ["X0","X1",…]
    elif len(feature_names) != n_features:
        # If passed count of feature names doesn't match number of columns
        raise FeatureNamesLengthError(len(feature_names), n_features)

    # Initialize internal structure
    internal: InternalState = {
        "parameters": {
            "verbose": kwargs.get("verbose", []),
            "approach": approach,
            "feature_names": list(feature_names),
            "n_explain": x_explain_arr.shape[0],
            "n_features": n_features,
            "n_MC_samples": kwargs.get(
                "n_MC_samples", 1000
            ),  # TODO (milanagm): what is being done here?
        },
        "data": {"x_train": x_train_arr, "x_explain": x_explain_arr},
        "iter_list": [{}],  # TODO (milanagm): do we need this and if so, for what?
        "timing_list": {},  # TODO (milanagm): do we need this and if so, for what?
    }

    # Select approach based on input
    if approach == "gaussian":
        approach_instance = GaussianApproach(internal)
    elif approach == "Copula":
        approach_instance = CopulaApproach(internal)
    else:
        error_msg = (
            f"Unknown approach: {approach}. "
            "Please use either 'gaussian' or 'Copula' as approach parameter."
        )
        raise ValueError(error_msg)

    internal = approach_instance.setup_approach()
    # TODO (milanagm): this method so far is not explaining anything but only calling setup_specific (mathod 1 in r script)

    return internal
