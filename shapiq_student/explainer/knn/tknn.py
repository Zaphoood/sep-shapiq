"""Implementation of TKNNExplainer Class."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import numpy as np
from shapiq import InteractionValues
from sklearn.neighbors import RadiusNeighborsClassifier

from .base import KNNExplainerBase, interactionvaluesfromarray

if TYPE_CHECKING:
    import numpy.typing as npt
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

        # Compute counting queries on full dataset D
        c = N  # Dataset size
        c_xval_tau = len(neighbor_indices) + 1  # Neighbors + validation point
        c_zval_tau = np.sum(self.y_train[neighbor_indices] == y_val)  # Same label neighbors

        # Handle no neighbors case (marginal contribution is 0)
        if c_xval_tau == 1:  # No training neighbors
            return interactionvaluesfromarray(sv)

        # Compute A2 (reusable sum)
        A2_sum = 0.0
        for k in range(c):
            if c - k > 0:  # Avoid division by zero
                binom_coeff = math.comb(c, k)
                numerator = c - k - c_xval_tau
                denominator = c - k

                if denominator != 0:
                    ratio = numerator / denominator
                    # Correct A2 formula: (1/(k+1)) * (1 - ratio) * binom_coeff / (2**c)
                    weighted_term = (1 / (k + 1)) * (1 - ratio) * binom_coeff / (2**c)
                    A2_sum += weighted_term

        # Subtract 1 from A2 as per paper's formula
        A2_sum = A2_sum - 1

        # Compute Shapley values for each neighbor using leave-one-out statistics
        for i in neighbor_indices:
            y_i = self.y_train[i]
            same_label = int(y_i == y_val)

            # Leave-one-out counting queries for point i (as per paper's C.2.2)
            c_xval_tau_leave_one_out = c_xval_tau - 1  # cx(val),τ(D\zi) = cx(val),τ(D) - 1[zi ∈ NB]
            c_zval_tau_leave_one_out = (
                c_zval_tau - same_label
            )  # c(+)z(val),τ(D\zi) = c(+)z(val),τ(D) - 1[zi ∈ NB] * 1[yi = y(val)]

            # Correct A1 computation using leave-one-out statistics
            A1_part1 = same_label / c_xval_tau_leave_one_out
            A1_part2 = (
                c_zval_tau_leave_one_out
                / (c_xval_tau_leave_one_out * (c_xval_tau_leave_one_out - 1))
                if c_xval_tau_leave_one_out > 1
                else 0.0
            )
            A1 = A1_part1 - A1_part2

            # Final formula using leave-one-out c_xval_tau as per paper
            sv[i] = (1 / (c_xval_tau_leave_one_out**2)) * (A1 - A2_sum) * same_label + (1 / self.C)

        # Set Shapley values for points outside the threshold to 0
        outside_mask = distances > self.tau
        sv[outside_mask] = 0.0

        return interactionvaluesfromarray(sv)
