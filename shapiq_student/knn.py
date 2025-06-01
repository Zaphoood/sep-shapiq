import shapiq
import numpy as np


class KNNShapley:
    def __init__(self, K=10, distance_fn=None):
        """
        Initialize the KNN Shapley Calculator.

        Parameters:
        - K (int): Number of Nearest Neighbours.
        - distance_fn (callable): Optional custom distance function. Defaults to Euclidean.
        """
        self.K = K
        self.distance_fn = distance_fn if distance_fn is not None else self._euclidean_distance
        self.support_values = None

    def _euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Compute support values for training data.

        Parameters:
        - X_train (np.ndarray): Training features, shape (N, d).
        - y_train (np.ndarray): Training labels, shape (N,).
        - X_test (np.ndarray): Test features, shape (N_test, d).
        - y_test (np.ndarray): Test labels, shape (N_test,).

        Returns:
        - np.ndarray: Shapley values for training data, shape (N,).
        """
        N = len(X_train)
        N_test = len(X_test)
        s = np.zeros(N)

        for j in range(N_test):
            x_test_j = X_test[j]
            y_test_j = y_test[j]

            distances = [self.distance_fn(x, x_test_j) for x in X_train]
            sorted_indices = np.argsort(distances)

            s_j = np.zeros(N)
            last_idx = sorted_indices[-1]
            s_j[last_idx] = int(y_train[last_idx] == y_test_j) / N

            for i in reversed(range(N - 1)):
                idx_i = sorted_indices[i]
                idx_ip1 = sorted_indices[i + 1]

                label_match_i = int(y_train[idx_i] == y_test_j)
                label_match_ip1 = int(y_train[idx_ip1] == y_test_j)

                term = (label_match_i - label_match_ip1) * min(self.K, i + 1) / (self.K * (i + 1))
                s_j[idx_i] = s_j[idx_ip1] + term

            s += s_j

        s /= N_test
        self.support_values = s
        return s
