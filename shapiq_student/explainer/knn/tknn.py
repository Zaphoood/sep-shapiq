"""TKNN Classifier Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import numpy as np
from scipy.special import comb

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq import InteractionValues
    from sklearn.neighbors import RadiusNeighborsClassifier


from .base import KNNExplainerBase, interaction_values_from_array


class TKNNExplainer(KNNExplainerBase):
    """TKNN Classifier Explainer.

    For calculating exact shapley values for a thresholded KNN Classifier.
    """

    def __init__(self, model: RadiusNeighborsClassifier, class_index: int) -> None:
        """Initialize TKNN Explainer.

        Args:
            model: RadiusNeighborsClassifier
            class_index: The class index to explain

        Raises:
            TypeError: If model is not RadiusNeighborsClassifier
        """
        super().__init__(model, class_index)
        self._model = model

        self.tau = cast("float", model.radius)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n_train = self.X_train.shape[0]
        n_classes = len(self.y_train_indices)

        neighbor_indices = self._model.radius_neighbors(x.reshape(1, -1), return_distance=False)
        neighbor_indices = neighbor_indices[0]

        in_neighborhood = np.zeros((n_train,), dtype=bool)
        in_neighborhood[neighbor_indices] = True

        y_train_is_class_index = self.y_train_indices == self.class_index

        # For entire dataset D
        c_D = n_train
        c_x_tau_D = 1 + np.sum(in_neighborhood)
        c_plus_z_tau_D = np.sum(in_neighborhood & y_train_is_class_index)

        # For each training point z_i
        c = c_D - 1
        c_x_tau = c_x_tau_D - in_neighborhood
        c_plus_z_tau = c_plus_z_tau_D - (in_neighborhood & y_train_is_class_index)

        a1 = np.zeros((n_train,), dtype=np.float64)
        a1[in_neighborhood] = y_train_is_class_index[in_neighborhood] / c_x_tau[
            in_neighborhood
        ] - c_plus_z_tau[in_neighborhood] / (
            c_x_tau[in_neighborhood] * (c_x_tau[in_neighborhood] - 1)
        )

        a2 = np.zeros((n_train,), dtype=np.float64)
        for i in range(n_train):
            if not in_neighborhood[i] or c_x_tau[i] < 2:  # noqa: PLR2004
                continue
            for k in range(c + 1):
                binom_term = comb(c - k, c_x_tau[i]) / comb(c + 1, c_x_tau[i])
                a2[i] += 1 / (k + 1) * (1 - binom_term) - 1

        second_summand = np.zeros((n_train,), dtype=np.float64)
        second_summand[in_neighborhood] = (
            y_train_is_class_index[in_neighborhood] - 1 / n_classes
        ) / c_x_tau[in_neighborhood]

        sv = a1 * a2 + second_summand

        return interaction_values_from_array(sv)
