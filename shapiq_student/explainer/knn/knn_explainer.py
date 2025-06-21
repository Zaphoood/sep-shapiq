"""KNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._base import KNNExplainerBase

if TYPE_CHECKING:
    import sklearn.neighbors


class KNNClassifierExplainer(KNNExplainerBase):
    """KNN Classifier Explainer.

    For calculating exact shapley values for a KNN Classifier.
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
        - model: KNN Classifier to be explained. Is used to call __knn_super to extract
          the training data from the model
        - class_index (int): The index of y_test to be explained. Defaults to 1.
        """
        self.data = data
        self.class_index = class_index
        self.model = model
        self.distance_fn = self._euclidean_distance
        self.shapley_values = None

        KNNExplainerBase.__init__(self.model)

        self.X_train = super.X_train
        self.y_train_indices = super.y_train_indices
        self.y_train_classes = super.y_train_classes
        self.K = super.k

    def explain_function(self, X_test: np.ndarray) -> np.ndarray:
        """Compute shapley values for training data.

        Parameters:
        - X_test (np.ndarray): Test features, shape (N_test, d).
        - class_index (int):  The index to be explained. Defaults to 1.

        Returns:
        - np.ndarray: Shapley values for training data, shape (N,).
        """
        self.X_test = X_test

        N = len(self.X_train)
        s = np.zeros(self.N)

        sorted_indices = self.model.kneighbors(
            X=[self.X_test], n_neighbors=N, return_distance=False
        )

        for i in range(N):
            if self.y_train_indices[sorted_indices[i]] == self.class_index:
                s[i] = 1 / N

        for j in reversed(range(N - 1)):
            if (self.y_train_indices[sorted_indices[j]] == self.class_index) and (
                self.y_train_indices[sorted_indices[j + 1]] == self.class_index
            ):
                s[j] = s[j + 1]
            elif (self.y_train_indices[sorted_indices[j]] == self.class_index) and (
                self.y_train_indices[sorted_indices[j + 1]] != self.class_index
            ):
                s[j] = s[j + 1] + (1 / self.K) * ((min(self.K, j)) / j)
            elif (self.y_train_indices[sorted_indices[j]] != self.class_index) and (
                self.y_train_indices[sorted_indices[j + 1]] == self.class_index
            ):
                s[j] = s[j + 1] - (1 / self.K) * ((min(self.K, j)) / j)
            else:
                s[j] = s[j + 1]

        return s
