"""Implementation of KNNExplainerBase class."""

from __future__ import annotations

import numpy as np
from shapiq import Explainer, InteractionValues
from sklearn.neighbors import KNeighborsClassifier

from ._base import KNNExplainerBase


class TKNNExplainer(KNNExplainerBase):
    """Threshold k-Nearest Neighbors Shapley explainer."""

    def __init__(self, model: KNeighborsClassifier, tau: float) -> None:
        """Init for TKNNExplainer.

        Args:
            model: KNeighborsClassifier
            tau: threshold

        Returns:
            TKNNExplainer object
        """
        super().__init__(model)
        message = "'tau' must be strictly positive."
        if tau <= 0:
            raise ValueError(message)

        self.tau = tau
        self.C = len(self.model.classes_)  # Number of classes

    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        """Compute TKNN-Shapley values for a given input point.

        Args:
            x: Input point to explain (will be reshaped to (1, -1))

        Returns:
            InteractionValues: Shapley values for each training point
        """
        x = x.reshape(1, -1)
        N = len(self.X_train)
        tau = self.tau

        # Get prediction for validation point
        y_val = self.model.predict(x)[0]

        # Initialize Shapley values
        sv = np.zeros(N)

        # TODO:: Compute distances and find neighbors (using < not <=, matching original)
        # TODO: use < OR <= 8 (original paper formula specifies <= HOWEVER implementation implements <)
        distances = np.array([np.linalg.norm(x_train_point - x) for x_train_point in self.X_train])
        neighbor_indices = (distances <= tau).nonzero()[0]  # RELEVANT PART HERE

        Ct = len(neighbor_indices)  # Number of neighbors

        if Ct == 0:
            # No neighbors - return zeros
            return InteractionValues(
                values=sv,
                index="SV",  # Use string identifier instead of list
                max_order=1,
                min_order=1,
                n_players=N,
                baseline_value=0.0,  # TODO: 1/C
            )

        # Count same-label neighbors
        Ca = np.sum(self.y_train[neighbor_indices] == y_val)

        # Compute reusable_sum exactly as in original implementation
        reusable_sum = 0
        stable_ratio = 1
        for j in range(N):
            if (N - j) > 0:  # Avoid division by zero
                stable_ratio *= (N - j - Ct) / (N - j)
                reusable_sum += (1 / (j + 1)) * (1 - stable_ratio)

        # Compute Shapley values for each neighbor
        for i in neighbor_indices:
            yi = self.y_train[i]

            # Base term: (indicator - uniform) / number of neighbors
            base_term = (int(yi == y_val) - 1 / self.C) / Ct
            sv[i] = base_term

            # Correction term (only if Ct >= 2)
            if Ct >= 2:
                ca = Ca - int(yi == y_val)  # Same-label neighbors excluding point i
                correction = (int(yi == y_val) / Ct - ca / (Ct * (Ct - 1))) * (reusable_sum - 1)
                sv[i] += correction

        # Return InteractionValues object
        return InteractionValues(
            values=sv,
            index="SV",  # Use string identifier for Shapley values
            max_order=1,
            min_order=1,
            n_players=N,
            baseline_value=0.0,  # TODO:should be 1/C??
        )


# Test the inheritance structure
def test_inheritance():
    """Test that the inheritance structure works correctly"""
    # Create test data
    np.random.seed(42)
    X = np.random.randn(10, 3)
    y = np.array([0, 1, 0, 1, 2, 0, 1, 2, 0, 1])
    x_test = np.array([0.5, 0.5, 0.5])

    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Test TKNN explainer
    tau = 2.0
    explainer = TKNNExplainer(knn, tau)

    print("INHERITANCE STRUCTURE TEST")
    print("=" * 40)
    print(
        f"TKNNExplainer isinstance of KNNExplainerBase: {isinstance(explainer, KNNExplainerBase)}"
    )
    print(f"TKNNExplainer isinstance of Explainer: {isinstance(explainer, Explainer)}")
    print(
        f"KNNExplainerBase is subclass of Explainer: {issubclass(explainer.__class__.__bases__[0], Explainer)}"
    )

    # Test the explain function returns InteractionValues
    result = explainer.explain(x_test)
    print(f"Result type: {type(result)}")
    print(f"Result isinstance of InteractionValues: {isinstance(result, InteractionValues)}")
    print(f"Shapley values: {result.values}")
    print(f"Sum of Shapley values: {np.sum(result.values):.6f}")

    # Test that it still matches the original implementation
    def tnn_shapley_single_original(
        x_train_few, y_train_few, x_test, y_test, tau=0, dis_metric="euclidean"
    ):
        N = len(y_train_few)
        sv = np.zeros(N)
        C = max(y_train_few) + 1

        if dis_metric == "cosine":
            distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
        else:
            distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

        Itau = (distance < tau).nonzero()[0]
        Ct = len(Itau)
        Ca = np.sum(y_train_few[Itau] == y_test)

        reusable_sum = 0
        stable_ratio = 1
        for j in range(N):
            if (N - j) > 0:
                stable_ratio *= (N - j - Ct) / (N - j)
                reusable_sum += (1 / (j + 1)) * (1 - stable_ratio)

        for i in Itau:
            sv[i] = (int(y_test == y_train_few[i]) - 1 / C) / Ct
            if Ct >= 2:
                ca = Ca - int(y_test == y_train_few[i])
                sv[i] += (int(y_test == y_train_few[i]) / Ct - ca / (Ct * (Ct - 1))) * (
                    reusable_sum - 1
                )

        return sv

    y_test_pred = knn.predict(x_test.reshape(1, -1))[0]
    original_result = tnn_shapley_single_original(X, y, x_test, y_test_pred, tau=tau)

    print("\nVerification against original:")
    print(f"Max difference: {np.max(np.abs(result.values - original_result)):.10f}")
    print(f"Results match: {np.allclose(result.values, original_result, atol=1e-10)}")


if __name__ == "__main__":
    test_inheritance()
