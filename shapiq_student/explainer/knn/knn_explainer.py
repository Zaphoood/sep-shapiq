"""KNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq_student.explainer.knn import KNNExplainerBase

if TYPE_CHECKING:
    import sklearn.neighbors


class KNNClassifierExplainer(KNNExplainerBase):
    """KNN Classifier Explainer.

    For calculating exact shapley values for an unweighted KNN Classifier.
    """

    def __init__(
        self,
        data: None,
        model: sklearn.neighbors.KNeighborsClassifier,
        class_index: int = 1,
    ) -> None:
        """Initialize the KNN Shapley Calculator.

        Parameters:
        - data (None): Not needed, only to fit the shap-iq structure.
        - model: KNN Classifier to be explained. Is used to call __knn_super to extract
          the training data from the model
        - class_index (int): The index of y_test to be explained. Defaults to 1.
        """
        super().__init__(model)
        self.data = data
        self.class_index = class_index

    def explain_function(self, X_test: np.ndarray) -> np.ndarray:
        """Compute shapley values for training data.

        Parameters:
        - X_test (np.ndarray): Test features, shape (N_test, d).
        - class_index (int): The class index to be explained. Defaults to 1.

        Returns:
        - np.ndarray: Shapley values for training data, shape (N,).
        """
        self.X_test = X_test

        N = len(self.X_train)
        s = np.zeros(N)

        sorted_indices_get = self.model.kneighbors(
            X=[self.X_test], n_neighbors=N, return_distance=False
        )
        sorted_indices = sorted_indices_get[0]

        for i in range(N):
            idx = sorted_indices[i]
            if self.y_train_indices[idx] == self.class_index:
                s[i] = 1 / N

        for j in reversed(range(N - 1)):
            idxj = sorted_indices[j]
            idxj_plusplus = sorted_indices[j + 1]
            if (self.y_train_indices[idxj] == self.class_index) and (
                self.y_train_indices[idxj_plusplus] == self.class_index
            ):
                s[j] = s[j + 1]
            elif (self.y_train_indices[idxj] == self.class_index) and (
                self.y_train_indices[idxj_plusplus] != self.class_index
            ):
                s[j] = s[j + 1] + (1 / self.k) * ((min(self.k, (j + 1))) / (j + 1))
            elif (self.y_train_indices[idxj] != self.class_index) and (
                self.y_train_indices[idxj_plusplus] == self.class_index
            ):
                s[j] = s[j + 1] - (1 / self.k) * ((min(self.k, (j + 1))) / (j + 1))
            else:
                s[j] = s[j + 1]

        backsort = sorted(zip(sorted_indices, s, strict=False))
        indices, backsorted_s = np.array(list(zip(*backsort, strict=False)))

        return backsorted_s
