"""KNN_Explainer - Wrapper for the Different KNN Explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shapiq

if TYPE_CHECKING:
    import numpy as np
    import sklearn.neighbors


class KNNExplainer(shapiq.Explainer):
    """KNN Explainer.

    Used for Extracting the training data from the model.
    """

    def __init__(
        self,
        model: sklearn.neighbors.KNeighborsClassifier,
    ) -> None:
        """Initialize the KNN Shapley Calculator.

        Parameters:
        - model: Type of the KNN Classifier to be explained.
        """
        self.model = model

        self.K: int = 5  # to be taken from the model
        self.X_train: np.ndarray  # to be taken from the model
        self.y_train: np.ndarray  # to be taken from the model

    def _get_y_train(self) -> np.ndarray:
        """Getting the y-values of the training data.

        Extracts the training data's y-values in a np.array.
        """
        return self.y_train

    def _get_x_train(self) -> np.ndarray:
        """Getting the X-values of the training data.

        Extracts the training data's X-values in a np.array.
        """
        return self.X_train

    def _get_k(self) -> int:
        """Getting K from the model.

        Extracts the number of nearest neighbours from the model.
        """
        return self.K
