"""KNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq_student.explainer.knn import KNNExplainerBase

if TYPE_CHECKING:
    import sklearn.neighbors

# TODO(Max): implement y_test structure as given on thursday.  Either implement multi-dimensional array handling or add information in the doc-strings, that only one-dimensional arrays are accepted.


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
        - data (None): Not used, only to fit the shap-iq structure.
        - model: KNN Classifier to be explained. Accepts a fitted instance of
            scikit-learn's KNeighborsClassifier.
        - class_index (int): The class-index of the classifier to be explained. Defaults to 1.
            The class index should be set. To explain more than one class, additional instances of KNNClassifierExplainer are needed.

        The KNNClassifierExplainer should not be calles directly but by using the shapiq.explainer.Explainer.
        """
        super().__init__(model)
        self.data = data
        self.class_index = class_index

    def explain_function(self, X_test: np.ndarray) -> np.ndarray:
        """Compute shapley values for training data.

        Parameters:
        - X_test (np.ndarray): Test features, shape (N_test, d).

        Returns:
        - np.ndarray: Shapley values for training data, shape (N,).

        Not to be used directly. Use shapiq's explain() instead. To calculate the shapley values for more than one data point use shapiq's explain_X().
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
