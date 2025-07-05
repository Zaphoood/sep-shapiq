"""KNN Classifier Explainer."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from shapiq_student.explainer.knn.util import keep_first_n

from .base import KNNExplainerBase, interaction_values_from_array
from .lookup_game import LookupGame

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq import InteractionValues
    import sklearn.neighbors


class BruteForceBasicKNNExplainer(KNNExplainerBase):
    """Brute force approach to computing Shapley Values for basic KNN models."""

    @override
    def __init__(
        self,
        model: sklearn.neighbors.KNeighborsClassifier,
        class_index: int,
    ) -> None:
        super().__init__(model, class_index)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        utilities = {}

        sortperm = self.model.kneighbors(
            x.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=False
        )
        sortperm = sortperm[0]
        y_train_sorted = self.y_train_indices[sortperm]

        for coalition_generator in product([False, True], repeat=self.X_train.shape[0]):
            coalition = np.array(list(coalition_generator))
            coalition_first_k = keep_first_n(coalition, n=self.k)
            utility = np.sum(y_train_sorted[coalition_first_k] == self.class_index) / self.k

            coalition_tuple = tuple(sorted(sortperm[coalition]))
            utilities[coalition_tuple] = utility

        game = LookupGame(n_players=self.X_train.shape[0], utilities=utilities)
        iv = game.exact_values("SII", order=1)

        return iv


class BasicKNNExplainer(KNNExplainerBase):
    """Explainer for basic KNN models.

    Efficiently calculates Shapley Values for unweighted k-Nearest-Neighbour models.
    """

    # TODO(Zaphoood): Explain functionality in class docstring

    @override
    def __init__(
        self,
        model: sklearn.neighbors.KNeighborsClassifier,
        class_index: int,
    ) -> None:
        super().__init__(model, class_index)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n = len(self.X_train)
        sv = np.zeros(n)

        sortperm = self.model.kneighbors(x.reshape(1, -1), n_neighbors=n, return_distance=False)
        sortperm = sortperm[0]

        y_train_indices_sorted = self.y_train_indices[sortperm]
        # Compute indicator function of whether a training point's class agrees with the class to explain
        y_train_is_class_index = (y_train_indices_sorted == self.class_index).astype(int)

        sv[-1] = y_train_is_class_index[-1] / n

        for i in range(n - 2, -1, -1):
            sv[i] = sv[i + 1] + (
                (y_train_is_class_index[i] - y_train_is_class_index[i + 1]) / self.k
            ) * (min(self.k, (i + 1)) / (i + 1))

        inv_sortperm = np.zeros_like(sortperm)
        inv_sortperm[sortperm] = np.arange(sortperm.shape[0])

        return interaction_values_from_array(sv[inv_sortperm])
