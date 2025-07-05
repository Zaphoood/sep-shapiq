"""Implementation of TKNNExplainer Class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

import numpy as np
from shapiq import InteractionValues
from sklearn.neighbors import RadiusNeighborsClassifier

from .base import KNNExplainerBase, interactionvaluesfromarray

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq import InteractionValues


class TKNNExplainerEfficient(KNNExplainerBase):
    """Threshold k-Nearest Neighbors Shapley explainer."""

    def __init__(
        self,
        model: RadiusNeighborsClassifier,
        class_index: int,
    ) -> None:
        """Initialize TKNNExplainer.

        Args:
            model: RadiusNeighborsClassifier only
            class_index: The class index to explain

        Raises:
            TypeError: If model is not RadiusNeighborsClassifier
        """
        if not isinstance(model, RadiusNeighborsClassifier):
            message = "Model must be RadiusNeighborsClassifier."
            raise TypeError(message)
        super().__init__(model, class_index)
        self.tau = model.radius  # Threshold
        self.C = len(self.model.classes_)  # Number of classes

    @override
    def explain_function(
        self, x: npt.NDArray[np.floating], *args: Any, **kwargs: Any
    ) -> InteractionValues:
        """Compute TKNN-Shapley values for a given input point.

        Args:
            x: Input point to explain (will be reshaped to (1, -1))
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            InteractionValues: Shapley values for each training point
        """
        x = x.reshape(1, -1)  # Point to explain (validation data point)
        N = len(self.X_train)  # Total number of training points

        # Get prediction for validation point
        y_val = self.model.predict(x)[0]

        # Initialize Shapley values
        sv = np.zeros(N)

        # Compute distances to all training points
        distances = np.array([np.linalg.norm(x_train_point - x) for x_train_point in self.X_train])

        # Find neighbors within threshold (<= as per paper)
        neighbor_indices = (distances <= self.tau).nonzero()[0]

        # Number of neighbors +1 (include validation point as per paper)
        C_t = len(neighbor_indices) + 1

        # If no neighbors (only validation point), return zeros
        if C_t == 1:
            return interactionvaluesfromarray(sv)

        # Count same-label neighbors
        C_a = np.sum(self.y_train[neighbor_indices] == y_val)

        # Compute reusable sum (efficient O(N) computation from paper)
        reusable_sum = 0.0
        stable_ratio = 1.0
        for j in range(N):
            if N - j > 0:  # Avoid division by zero
                stable_ratio *= (N - j - C_t) / (N - j)
                reusable_sum += (1 / (j + 1)) * (1 - stable_ratio)

        # Compute Shapley values for each neighbor
        for i in neighbor_indices:
            y_i = self.y_train[i]
            same_label = int(y_i == y_val)

            # Base term: (indicator - 1/C) / C_t (corrected from reference implementation)
            base_term = (same_label - 1 / self.C) / C_t
            sv[i] = base_term

            comparison = 2

            # Correction term (only if C_t >= 2)
            if C_t >= comparison:
                c_a = C_a - same_label  # Leave-one-out same-label count
                correction = (same_label * c_a) / (C_t * (C_t - 1))
                sv[i] += correction * (reusable_sum - 1)

        # Points outside threshold remain 0 (already initialized)
        return interactionvaluesfromarray(sv)
