"""Abstract base class for imputation approaches.

This module defines the base class for different Gaussian-based imputation approaches used in SHAP value
calculations. It inherits from shapiq's Imputer base class and provides a common interface that all specific approaches must implement.
"""

# TODO(Zaphoood): Make sure all docstrings follow Google's style

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .exceptions import EmptyDataError

if TYPE_CHECKING:
    import numpy.typing as npt

from shapiq.games.imputer.base import Imputer


class GaussianImputerBase(Imputer):
    """Abstract base class for Gaussian-based imputation approaches.

    This class inherits from shapiq's Imputer base class and defines the interface that all specific
    Gaussian-based imputation approaches must implement.
    """

    def __init__(
        self,
        model: object,
        data: npt.NDArray[np.floating],
        x: npt.NDArray[np.floating] | None = None,
        *,
        n_mc_samples: int = 1000,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes GaussianImputerBase.

        Args:
            model (object): The model to explain as a callable function expecting data points as input and
                returning the model's predictions.
            data (npt.NDArray[np.floating]): The background data to use for the explainer as a 2-dimensional array with shape
                (n_samples, n_features).
            x (npt.NDArray[np.floating] | None, optional): The explanation point to use the imputer on either as a 2-dimensional array with
                shape (1, n_features) or as a vector with shape (n_features,). Defaults to None.
            n_mc_samples (int, optional): Number of Monte Carlo samples for imputation. Defaults to 1000.
            random_state (int | None, optional): The random state to use for sampling. Defaults to None.
            verbose (bool, optional): A flag to enable verbose imputation, which will print a progress bar for model
                evaluation. Note that this can slow down the imputation process. Defaults to False.

        Raises:
            EmptyDataError: If the provided data is empty.
        """
        if data is None or np.size(data) == 0 or (hasattr(data, "shape") and data.shape[0] == 0):
            raise EmptyDataError
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=n_mc_samples,
            categorical_features=[],
            random_state=random_state,
            verbose=verbose,
        )

        self.n_mc_samples = n_mc_samples
        self._mean_per_feature: npt.NDArray[np.floating] | None = None
        self._cov_mat: npt.NDArray[np.floating] | None = None

    @property
    def mean_per_feature(self) -> npt.NDArray[np.floating]:
        """Get the mean values per feature, computing them if not already computed.

        Returns:
            npt.NDArray[np.floating]: The mean value for each feature.
        """
        if self._mean_per_feature is None:
            self._mean_per_feature = np.mean(self.data, axis=0)
        return self._mean_per_feature

    @property
    def cov_mat(self) -> npt.NDArray[np.floating]:
        """Get the covariance matrix, computing it if not already computed.

        Returns:
            npt.NDArray[np.floating]: The covariance matrix of the data.
        """
        if self._cov_mat is None:
            self._cov_mat = self._ensure_positive_definite(np.cov(self.data.T))
        return self._cov_mat

    def _ensure_positive_definite(
        self,
        cov_mat: npt.NDArray[np.floating],
        min_eigen_value: float = 1e-06,
    ) -> npt.NDArray[np.floating]:
        """Ensure covariance matrix is positive definite by correcting eigenvalues if necessary.

        Args:
            cov_mat (npt.NDArray[np.floating]): Input covariance matrix.
            min_eigen_value (float, optional): Minimum allowed eigenvalue. Defaults to 1e-06.

        Returns:
            npt.NDArray[np.floating]: Positive definite covariance matrix.
        """
        eigen_values = np.linalg.eigvalsh(cov_mat)

        # If any eigenvalue is too small (close to zero or negative)
        if np.any(eigen_values <= min_eigen_value):
            # Add regularization to make it positive definite
            min_actual_eigen_value = np.min(eigen_values)
            cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * (
                min_eigen_value - min_actual_eigen_value
            )

        return cov_mat
