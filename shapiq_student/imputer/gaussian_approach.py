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
MAX_UNIQUE_VALUES_FOR_CATEGORICAL = 10  # randomly assigned


class GaussianApproach(Approach):
    """Implementation of Gaussian-based approach for SHAP value calculations.

    This approach uses multivariate normal distribution for generating samples.
    It only supports continuous features and raises an error if categorical features
    are detected.
    """

    def _check_factor_features(self) -> None:
        """Check if any features are categorical/factor variables.

        Raises:
        ------
        ValueError
            If any categorical features are detected.
        """
        x_train = self.internal.get("data", {}).get("x_train")
        feature_names = self.internal.get("parameters", {}).get("feature_names", [])
        dtypes = x_train.dtype
        factor_features = []
        for i, feature in enumerate(x_train.T):
            if np.issubdtype(dtypes[i], np.integer):
                unique_values = np.unique(feature)
                if len(unique_values) < MAX_UNIQUE_VALUES_FOR_CATEGORICAL:
                    factor_features.append(feature_names[i])
            elif np.issubdtype(dtypes[i], np.object_):
                factor_features.append(feature_names[i])
        if factor_features:
            msg = (
                f"The following are categorical/factor features: {', '.join(factor_features)}. "
                "Gaussian approach does not support categorical features."
            )
            raise ValueError(msg)

    def calculate_mean_per_feature(self) -> None:
        """Calculate the mean value for each feature in the training data.

        This method computes the mean value for each feature (column) in the training data
        and stores it in the internal parameters dictionary.

        Raises:
        ------
        TypeError
            If training data is not a numpy array.
        ValueError
            If training data is empty.
        """
        x_train = self.internal.get("data", {}).get("x_train")
        if not isinstance(x_train, np.ndarray):
            msg = "Training data must be a numpy array."
            raise TypeError(msg)
        if x_train.size == 0:
            msg = "Training data is empty."
            raise ValueError(msg)
        if "mean_per_feature" not in self.internal["parameters"]:
            self.internal["parameters"]["mean_per_feature"] = np.mean(x_train, axis=0)

    def calculate_covariance_matrix(self) -> None:
        """Compute the covariance matrix of the training data.

        This method calculates the covariance matrix from the training data
        and stores it in the internal parameters dictionary.

        Raises:
        ------
        TypeError
            If training data is not a numpy array.
        ValueError
            If training data is empty.
        """
        x_train = self.internal.get("data", {}).get("x_train")
        if not isinstance(x_train, np.ndarray):
            msg = "Training data must be a numpy array."
            raise TypeError(msg)
        if x_train.size == 0:
            msg = "Training data is empty."
            raise ValueError(msg)
        if "cov_mat" not in self.internal["parameters"]:
            self.internal["parameters"]["cov_mat"] = np.cov(x_train.T)

    def _setup_specific(self) -> None:
        # (lironaisn): Instead of calling one method, we devide it into more methods so they can be tested seperately
        self._check_factor_features()
        self.calculate_mean_per_feature()
        self.calculate_covariance_matrix()

    def prepare_data(self, index_features: list[int] | None = None) -> dict[str, Any]:
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
        n_MC_samples = self.internal["parameters"]["n_MC_samples"]
        x_explain_mat = np.array(self.internal["data"]["x_explain"])
        mean_per_feature = self.internal["parameters"]["mean_per_feature"]
        cov_mat = self.internal["parameters"]["cov_mat"]

        # Generate MC samples from N(0,1)
        rng: Generator = default_rng()
        MC_samples_mat = rng.normal(size=(n_MC_samples, n_features))

        # Convert to N(mean_per_feature_{Sbar|S}, Sigma_{Sbar|S})
        # Note: This will raise NotImplementedError until implemented
        self._prepare_gaussian_data(
            MC_samples_mat=MC_samples_mat,
            x_explain_mat=x_explain_mat,
            S=self.internal["iter_list"][-1]["S"][index_features],
            mean_per_feature=mean_per_feature,
            cov_mat=cov_mat,
        )

        return {}

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
