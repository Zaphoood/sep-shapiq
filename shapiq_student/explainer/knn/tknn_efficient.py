"""Implement TKNN Efficient Explainer Class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

import numpy as np
from shapiq import InteractionValues
from sklearn.neighbors import RadiusNeighborsClassifier

from .base import KNNExplainerBase, interaction_values_from_array

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq import InteractionValues

# TODO @murscht: add docstrings


class TKNNExplainerEfficient(KNNExplainerBase):
    """Threshold k-Nearest Neighbors Shapley explainer (O(N), closed-form as in the paper)."""

    def __init__(self, model: RadiusNeighborsClassifier, class_index: int) -> None:
        """Initializes the efficient closed-form TKNNExplainer.

        Args:
            model: RadiusNeighborsClassifier
            class_index: The class index to explain
        Raises:
            TypeError: If model is not RadiusNeighborsClassifier
        """
        if not isinstance(model, RadiusNeighborsClassifier):
            model_error = "Model must be RadiusNeighborsClassifier."
            raise TypeError(model_error)
        super().__init__(model, class_index)
        self.tau = model.radius
        self.C = len(self.model.classes_)

    @override
    def explain_function(
        self, x: npt.NDArray[np.floating], *args: Any, **kwargs: Any
    ) -> InteractionValues:
        """Compute TKNN Shapley values for a given input point.

        Args:
            x: Input point to explain
            *args: Additional Arguments
            **kwargs: Additional Keyword Arguments
        Returns:
            InteractionValues: Shapley Values for each training point
        """
        x = x.reshape(1, -1)
        N = len(self.X_train)
        y_val = self.model.predict(x)[0]
        sv = np.zeros(N)

        # Compute distances to all training points
        distances = np.array([np.linalg.norm(x_train_point - x) for x_train_point in self.X_train])
        neighbor_indices = np.where(distances <= self.tau)[0]

        C_t = len(neighbor_indices) + 1  # neighbors + validation point
        if C_t == 1:
            return interaction_values_from_array(sv)

        C_a = np.sum(self.y_train[neighbor_indices] == y_val)

        # Compute reusable sum (A2) as in the paper, efficient O(N)
        reusable_sum = 0.0
        stable_ratio = 1.0
        for j in range(N):
            stable_ratio *= (N - j - C_t) / (N - j)
            reusable_sum += (1 / (j + 1)) * (1 - stable_ratio)
        A2 = reusable_sum - 1

        # Closed-form as in the paper, Appendix C.2.2
        for i in neighbor_indices:
            y_i = self.y_train[i]
            same_label = int(y_i == y_val)
            ca = C_a - same_label
            comparison = 2
            if C_t >= comparison:
                sv[i] = (same_label - 1 / self.C) / C_t + (
                    same_label / C_t - ca / (C_t * (C_t - 1))
                ) * A2
            else:
                sv[i] = (same_label - 1 / self.C) / C_t

        return interaction_values_from_array(sv)
