"""KNN Classifier Explainer."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
from shapiq import Game, InteractionValues

from shapiq_student.explainer.knn.base import interaction_values_from_array
from shapiq_student.explainer.knn.knn import KNNExplainerBase

if TYPE_CHECKING:
    import numpy.typing as npt
    import sklearn.neighbors


class LookupGame(Game):
    """Defines a Game via a dictionary giving the utility function."""

    def __init__(self, n_players: int, utilities: dict[tuple[int, ...], float]) -> None:
        """Initializes the LookupGame."""
        self.characteristic_function = utilities
        super().__init__(
            n_players=n_players,
            normalization_value=self.characteristic_function[()],
        )

    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        """Defines the worth of a coalition as a lookup in the characteristic function.

        Args:
            coalitions: A 2D array where each row represents a coalition as a binary
                vector (1 for present, 0 for absent).

        Returns:
            A 1D array containing the value of each coalition based on the
                characteristic function.
        """
        output = [
            self.characteristic_function[tuple(np.where(coalition)[0])] for coalition in coalitions
        ]
        return np.array(output)


class BruteForceKNNClassifierExplainer(KNNExplainerBase):
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
            coalition_first_k = _first_n_true(coalition, n=self.k)
            utility = np.sum(y_train_sorted[coalition_first_k] == self.class_index) / self.k

            coalition_tuple = tuple(sorted(sortperm[coalition]))
            utilities[coalition_tuple] = utility

        game = LookupGame(n_players=self.X_train.shape[0], utilities=utilities)
        iv = game.exact_values("SII", order=1)

        return iv


def _first_n_true(mask: npt.NDArray[np.bool], n: int) -> npt.NDArray[np.bool]:
    """Set all but the first n True entries of the given boolean mask to False.

    This will just return a reference to the input array if ``np.sum(mask) <= n``

    Args:
        mask: The mask in question.
        n: The maximum number of true entries.
    """
    if n == 0:
        return np.zeros_like(mask)

    n_true = 0
    for i, val in enumerate(mask):
        n_true += val
        if n_true == n:
            out = np.zeros_like(mask)
            out[: i + 1] = mask[: i + 1]
            return out

    return mask


class KNNClassifierExplainer(KNNExplainerBase):
    """KNN Classifier Explainer.

    For calculating exact shapley values for an unweighted KNN Classifier.
    """

    def __init__(
        self,
        model: sklearn.neighbors.KNeighborsClassifier,
        class_index: int,
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
        super().__init__(model, class_index)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n = len(self.X_train)
        sv = np.zeros(n)

        sortperm = self.model.kneighbors(x.reshape(1, -1), n_neighbors=n, return_distance=False)
        sortperm = sortperm[0]

        y_train_sorted = self.y_train_indices[sortperm]
        y_train_is_class_index = y_train_sorted == self.class_index

        sv[-1] = bool(y_train_is_class_index[-1]) / n

        for i in range(n - 2, -1, -1):
            sv[i] = sv[i + 1] + (
                (bool(y_train_is_class_index[i]) - bool(y_train_is_class_index[i + 1])) / self.k
            ) * (min(self.k, (i + 1)) / (i + 1))

        inv_sortperm = np.zeros_like(sortperm)
        inv_sortperm[sortperm] = np.arange(sortperm.shape[0])

        return interaction_values_from_array(sv[inv_sortperm])
