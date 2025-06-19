"""KNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knn_super import KNNExplainer
import numpy as np

if TYPE_CHECKING:
    import sklearn.neighbors


class KNNClassifierExplainer(KNNExplainer):
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
        self.class_index = class_index
        self.model = model
        self.data = data
        self.distance_fn = self._euclidean_distance
        self.support_values = None

        self.y_test: np.ndarray = None  # TODO(max): extract from model
        self.X_train: np.ndarray = None  # TODO(max): extract from model
        self.y_train: np.ndarray = None  # TODO(max): extract from model
        self.K: int = 5  # TODO(max): extract from model

    def _euclidean_distance(self, x1: any, x2: any) -> any:
        return np.linalg.norm(x1 - x2)

    def explain(self, X_test: np.ndarray) -> np.ndarray:
        """Compute shapley values for training data.

        Parameters:
        - X_test (np.ndarray): Test features, shape (N_test, d).
        - class_index (int):  Defaults to 1.

        Returns:
        - np.ndarray: Shapley values for training data, shape (N,).
        """
        self.X_test = X_test

        N = len(self.X_train)
        N_test = len(self.X_test)
        s = np.zeros(self.N)

        for j in range(N_test):
            x_test_j = X_test[j]
            y_test_j = self.y_test[j]

            distances = [self.distance_fn(x, x_test_j) for x in self.X_train]
            sorted_indices = np.argsort(distances)

            s_j = np.zeros(N)
            last_idx = sorted_indices[-1]
            s_j[last_idx] = int(self.y_train[last_idx] == y_test_j) / N

            for i in reversed(range(N - 1)):
                idx_i = sorted_indices[i]
                idx_ip1 = sorted_indices[i + 1]

                label_match_i = int(self.y_train[idx_i] == y_test_j)
                label_match_ip1 = int(self.y_train[idx_ip1] == y_test_j)

                term = (label_match_i - label_match_ip1) * min(self.K, i + 1) / (self.K * (i + 1))
                s_j[idx_i] = s_j[idx_ip1] + term

            s += s_j

        s /= N_test
        self.support_values = s
        return s
