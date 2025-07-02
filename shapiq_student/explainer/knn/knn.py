"""KNN Classifier Explainer."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np
from shapiq import Game, InteractionValues

from shapiq_student.explainer.knn import KNNExplainerBase
from shapiq_student.explainer.knn.base import interaction_lookup_from_knn_shapley_values

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
        class_index: int | None = None,
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

    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        """Compute shapley values for training data.

        Parameters:
        - X_test (np.ndarray): Test features, shape (N_test, d).

        Returns:
        - np.ndarray: Shapley values for training data, shape (N,).

        Not to be used directly. Use shapiq's explain() instead. To calculate the shapley values for more than one data point use shapiq's explain_X().
        """
        N = len(self.X_train)
        sv = np.zeros(N)

        sortperm = self.model.kneighbors(x.reshape(1, -1), n_neighbors=N, return_distance=False)
        sortperm = sortperm[0]

        if sortperm[-1] == self.class_index:
            sv[-1] = 1 / N

        for i in reversed(range(N - 1)):
            idxi = sortperm[i]
            idxi_plus = sortperm[i + 1]
            if (self.y_train_indices[idxi] == self.class_index) and (
                self.y_train_indices[idxi_plus] == self.class_index
            ):
                sv[i] = sv[i + 1]
            elif (self.y_train_indices[idxi] == self.class_index) and (
                self.y_train_indices[idxi_plus] != self.class_index
            ):
                sv[i] = sv[i + 1] + (1 / self.k) * ((min(self.k, (i + 1))) / (i + 1))
            elif (self.y_train_indices[idxi] != self.class_index) and (
                self.y_train_indices[idxi_plus] == self.class_index
            ):
                sv[i] = sv[i + 1] - (1 / self.k) * ((min(self.k, (i + 1))) / (i + 1))
            else:
                sv[i] = sv[i + 1]

        inv_sortperm = sorted(zip(sortperm, sv, strict=False))
        _, sv_backsorted = np.array(list(zip(*inv_sortperm, strict=False)))

        return interaction_lookup_from_knn_shapley_values(sv_backsorted)
