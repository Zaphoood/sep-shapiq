"""KNN_Explainer - Wrapper for the Different KNN Explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shapiq

if TYPE_CHECKING:
    import numpy as np
    import sklearn.neighbors


class KNNExplainer(shapiq.Explainer):
    """KNN Explainer.

    Wrapper class for KNNClassifierExplainer, KNNThresholdExplainer
    and WeightedKNNExplainer.
    Used for handling the model and the interface to shapiq.
    """

    def __init__(
        self,
        data: None,
        model: sklearn.neighbors.KNeighborsClassifier,
        class_index: int = 1,
    ) -> None:
        """Initialize the KNN Shapley Calculator.

        Parameters:
        - data (None): Not needed, only to fit the shap-iq structure. Defaults to None.
        - model: Type of the KNN Classifier to be explained.
        - class_index (int): Not needed, only to fit the shap-iq structure. Defaults to 1.
        """
        self.class_index = class_index
        self.model = model
        self.data = data
        self.distance_fn = self._euclidean_distance
        self.support_values = None

        self.K: int = 5  # to be taken from the model
        self.y_test: np.ndarray  # to be taken from the model
        self.X_train: np.ndarray  # to be taken from the model
        self.y_train: np.ndarray  # to be taken from the model

    def explain(self, X_test: np.ndarray) -> np.ndarray:
        """Handling the explain-Funktion from shapiqs Explainer.

        Explain is called by shapiq.Explainer.explain() and will be used to call the .explain()
        of the selected explainer.
        """
        return self.KNNClassifierExplainer.explain(
            X_test, self.y_testy, self.X_train, self.y_train, self.K
        )

    """
    def predict(x)
        to be added - should return the prediction of the chosen KNN-explainer.
    """
