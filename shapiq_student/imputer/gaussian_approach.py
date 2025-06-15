"""Gaussian approach for imputation in SHAP value calculations.

This module implements a Gaussian-based approach for generating samples in SHAP value
calculations. It uses multivariate normal distribution for sampling and handles
continuous features only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from numpy.random import Generator, default_rng

if TYPE_CHECKING:
    from numpy.typing import NDArray
from .approach import Approach

# Constants
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = (
    2  # Below see reasoning for selecting 3
    # Features with fewer than 3 unique values are likely either:
    # 1. categorical values as strings before beeing one-hot encoded
    # 2. Binary features (0/1)
    # 2. One-hot encoded features (which would have at most 2 values per encoded column)
)


class GaussianApproach(Approach):
    """Implementation of Gaussian-based approach for SHAP value calculations.

    This approach uses multivariate normal distribution for generating samples.
    It only supports continuous features and raises an error if categorical features
    are detected.
    """

    def _check_factor_features(self) -> None:  # TODO (milanagm): we need to add tests
        """Check if any features are categorical/factor variables.

        Raises:
        ------
        ValueError
            If any categorical features are detected.
        """
        x_train = self.internal["data"]["x_train"]
        feature_names = self.internal["parameters"]["feature_names"]

        # Get data types of each feature
        dtypes = x_train.dtype  # TODO (milanagm): dtype is callable on numpy arrays - should we adjust it to adress other formats too?

        factor_features = []

        for i, feature in enumerate(x_train.T):
            # Check for integer object type
            if np.issubdtype(dtypes[i], np.integer):
                # np.unique() returns all unique values in the feature
                # If there are fewer than threshold, we consider it categorical
                unique_values = np.unique(feature)
                if len(unique_values) <= MAX_UNIQUE_VALUES_FOR_CATEGORICAL:
                    factor_features.append(feature_names[i])

            # Check if feature is string/object type
            elif np.issubdtype(dtypes[i], np.object_):
                factor_features.append(feature_names[i])

        if factor_features:
            error_msg = (
                f"The following are categorical/factor features: {', '.join(factor_features)}. "
                "Gaussian approach does not support categorical features."
            )
            raise ValueError(error_msg)

    def _setup_specific(self) -> None:  # TODO (milanagm): we need to add tests
        """Check feature types and calculate mean and covariance matrix.

        This method checks for categorical features and initializes the mean
        and covariance matrix for the Gaussian approach.
        """
        # Check for factor features
        self._check_factor_features()  # TODO (milanagm): do we need to add checks for missing values?

        # Initialize mean for each column/feature in the training data if not provided
        if (
            "mean_per_feature" not in self.internal["parameters"]
        ):  # TODO (milanagm): we need to add tests
            self.internal["parameters"]["mean_per_feature"] = np.mean(
                self.internal["data"]["x_train"], axis=0
            )

        # Initialize covariance matrix if not provided
        if "cov_mat" not in self.internal["parameters"]:  # TODO (milanagm): we need to add tests
            self.internal["parameters"]["cov_mat"] = np.cov(self.internal["data"]["x_train"].T)

    def prepare_data(
        self, index_features: list[int] | None = None
    ) -> dict[
        str, Any
    ]:  # TODO (milanagm): check if we really need to set index_features= None #index features sind die indices die wir haben wollen # nochmal ansehen wie das passiert
        """Prepare data for Gaussian approach.

        Parameters
        ----------
        index_features : list[int] | None, optional
            List of feature indices to consider, by default None.

        Returns:
        -------
        dict[str, Any]
            Dictionary containing the prepared data.
        """
        n_features = self.internal["parameters"]["n_features"]
        n_MC_samples = self.internal["parameters"][
            "n_MC_samples"
        ]  # TODO (milanagm): check how this variable is set and compare to approach_gasussian.R
        x_explain_mat = np.array(self.internal["data"]["x_explain"])
        mean_per_feature = self.internal["parameters"]["mean_per_feature"]
        cov_mat = self.internal["parameters"]["cov_mat"]
        # TODO (milanagm): in approach_gasussian.R we also have n_coalitions_now set as index_features. we need to check where it is from and if we need it

        # TODO (milanagm): right now we are ignoring casual shapley values case. myb we need to add it still # keep it like that

        # Generate MC samples from N(0,1)
        rng: Generator = default_rng()
        MC_samples_mat = rng.normal(  # TODO (milanagm): check if method is correct and compare with in approach_gasussian.R
            size=(n_MC_samples, n_features)
        )

        # Convert to N(mean_per_feature_{Sbar|S}, Sigma_{Sbar|S})
        # Note: This will raise NotImplementedError until implemented
        self._prepare_gaussian_data(  # TODO (milanagm): check if method is correct and compare with in approach_gasussian.R # Guassian.cpp in python übertragen
            MC_samples_mat=MC_samples_mat,
            x_explain_mat=x_explain_mat,
            S=self.internal["iter_list"][-1]["S"][index_features],
            mean_per_feature=mean_per_feature,
            cov_mat=cov_mat,
        )

        return {}  # TODO (milanagm): Implement actual return value when _prepare_gaussian_data is implemented

    def _prepare_gaussian_data(
        self,
        MC_samples_mat: NDArray[np.float64],
        x_explain_mat: NDArray[np.float64],
        S: NDArray[np.float64],
        mean_per_feature: NDArray[np.float64],
        cov_mat: NDArray[np.float64],
    ) -> NoReturn:
        """Prepare Gaussian data for sampling.

        Parameters
        ----------
        MC_samples_mat : NDArray[np.float64]
            Matrix of Monte Carlo samples.
        x_explain_mat : NDArray[np.float64]
            Matrix of data to be explained.
        S : NDArray[np.float64]
            Feature set matrix.
        mean_per_feature : NDArray[np.float64]
            Mean vector for features.
        cov_mat : NDArray[np.float64]
            Covariance matrix.

        Raises:
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        error_msg = (
            "The Gaussian data preparation is not yet implemented. "
            "This method needs to be implemented according to the Gaussian.cpp logic."
        )
        raise NotImplementedError(error_msg)
