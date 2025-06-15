"""KNN_Explainer - Wrapper for the Different KNN Explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq import Explainer

if TYPE_CHECKING:
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier


class KNNExplainerBase(Explainer):
    """KNN Explainer.

    Used for Extracting the training data from the model.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
    ) -> None:
        """Initialize the KNN Shapley Calculator.

        Parameters:
        - model: Type of the KNN Classifier to be explained.
        """
        self.model = model

        raise NotImplementedError

        self.k: int = 5  # to be taken from the model
        self.X_train: np.ndarray  # to be taken from the model
        self.y_train: np.ndarray  # to be taken from the model
