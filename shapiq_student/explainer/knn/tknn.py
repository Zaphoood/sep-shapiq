"""Implementation of TKNNExplainer Class."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import numpy as np
from shapiq import InteractionValues
from sklearn.neighbors import RadiusNeighborsClassifier

from shapiq_student.explainer.knn import KNNExplainerBase, interactionvaluesfromarray

if TYPE_CHECKING:
    from shapiq import InteractionValues


class TKNNExplainer(KNNExplainerBase):
    """Threshold k-Nearest Neighbors Shapley explainer."""

    def __init__(self, model: RadiusNeighborsClassifier, class_index: int) -> None:
        """Initialize TKNNExplainer.

        Args:
            model: RadiusNeighborsClassifier only
            class_index: The class index to explain

        Raises:
            TypeError: If model is not RadiusNeighborsClassifier
        """
        if not isinstance(model, RadiusNeighborsClassifier):
            model_error_message = "Model must be RadiusNeighborsClassifier."
            raise TypeError(model_error_message)
        super().__init__(model, class_index)
        self.tau: float = model.radius  # use the radius from model as threshold tau
        self.C: int = len(self.model.classes_)  # number of classes

    @override
    def explain_function(self, x: np.ndarray, *args: Any, **kwargs: Any) -> InteractionValues:
        """Compute TKNN-Shapley values for a given input point.

        Args:
            x: Input point to explain (will be reshaped to (1, -1))
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            InteractionValues: Shapley values for each training point
        """
        x = x.reshape(1, -1)
        N = len(self.X_train)

        # Get prediction for validation point
        y_val = self.model.predict(x)[0]

        # Initialize Shapley values
        sv = np.zeros(N)

        # Compute distances to all training points
        distances = np.array([np.linalg.norm(x_train_point - x) for x_train_point in self.X_train])

        # Find neighbors within threshold
        neighbor_mask = distances <= self.tau
        neighbor_indices = np.where(neighbor_mask)[0]

        # Compute number of neighbors +1 (vor validation point)
        c_xval_tau = len(neighbor_indices) + 1

        # Handle no neighbors case (marginal contribution is 0)
        if c_xval_tau == 1:  # No training neighbors
            return interactionvaluesfromarray(sv)

        c_zval_tau = np.sum(
            self.y_train[neighbor_indices] == y_val
        )  # Number of same label neighbors
        c = N  # Dataset size
        A2_sum = 0.0

        for k in range(c + 1):
            if k + 1 > 0:  # Avoid division by zero
                term1 = 1.0 / (k + 1)

                if c_xval_tau > 0 and k <= c:
                    binom_coeff = math.comb(c, k)
                    numerator = (c - k) / c_xval_tau
                    denominator = (c + 1) / c_xval_tau

                    if denominator != 0:
                        ratio = numerator / denominator
                        term2 = term1 * ratio
                        weighted_term = binom_coeff * (term1 - term2) / (2**c)
                        A2_sum += weighted_term

        for i in neighbor_indices:
            y_i = self.y_train[i]
            same_label = int(y_i == y_val)
            A1_part1 = same_label / c_xval_tau
            A1_part2 = c_zval_tau / (c_xval_tau * (c_xval_tau - 1)) if c_xval_tau > 1 else 0.0
            A1 = A1_part1 - A1_part2
            sv[i] = (1 / (c_xval_tau**2)) * (A1 - A2_sum) * same_label + (1 / self.C)

        outside_mask = distances > self.tau
        sv[outside_mask] = 0.0

        return interactionvaluesfromarray(sv)
