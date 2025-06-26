"""Implementation of the explainer for weighted KNN models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

from .base import KNNExplainerBase, interaction_values_from_knn_shapley_values

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq.interaction_values import InteractionValues
    from sklearn.neighbors import KNeighborsClassifier

from itertools import product

import numpy as np
from shapiq.games import Game
from sklearn.neighbors._base import _get_weights as sklearn_get_weights


class ArrayGame(Game):
    """Game defined by an array containing the utility of each coalition."""

    def __init__(self, n_players: int, utility: npt.NDArray[np.floating]) -> None:
        """Initializes the ArrayGame.

        Args:
            n_players: The number of players.
            utility: An array of size 2**n_players that defines the utilitity function v, such that for some coalition S = {i_1, ..., i_n},
                v(S) == utility[2**i_1 + ... + 2**i_n].
        """
        if 2**n_players != utility.shape[0]:
            msg = "Size of utility array must be 2**n_players"
            raise ValueError(msg)

        self.utility = utility
        self._players = np.arange(n_players)

        super().__init__(n_players=n_players, normalization_value=self.utility[0])

    @override
    def value_function(self, coalitions: npt.NDArray[np.bool]) -> npt.NDArray[np.floating]:
        bases = np.repeat(2 ** np.arange(self.n_players)[None, :], coalitions.shape[0], axis=0)
        bases *= coalitions
        indices = cast("npt.NDArray[np.integer]", np.sum(bases, axis=1))

        return self.utility[indices]


class BruteForceWKNNExplainer(KNNExplainerBase):
    """A brute force implementation of WKNN according to `Wang et. al (2024)` [Wng24]_.

    References:
        .. [Wng24] Wang, Jiachen T., Prateek Mittal, and Ruoxi Jia. "Efficient data shapley for weighted nearest neighbor algorithms." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int,
    ) -> None:
        """Initializes the BruteForceWKNNExplainer.

        Args:
            model: The KNN model to explain.
            class_index: The class index of the model to explain.
        """
        super().__init__(model, class_index)

        model_weights = self.model.weights  # type: ignore[attr-defined]
        if model_weights != "distance":
            msg = f"KNeighboursClassifier must use weights='distance', but has weights='{model_weights}'"
            raise ValueError(msg)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        sortperm, weights = self._get_training_data_weights_sorted(x)
        n_players = self.X_train.shape[0]
        sv = np.zeros((n_players,))
        n_classes = len(self.y_train_classes)

        if n_classes == 1:
            return interaction_values_from_knn_shapley_values(
                np.zeros((n_players,), dtype=np.floating)
            )

        for other_class in self.y_train_classes:
            if other_class == self.class_index:
                continue
            sv_current = self._explain_binary(self.class_index, other_class, sortperm, weights)
            sv += sv_current

        sv /= n_classes - 1

        return interaction_values_from_knn_shapley_values(sv)

    def _explain_binary(
        self,
        y_val: int,
        y_other: int,
        sortperm: npt.NDArray[np.integer],
        weights: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        n_players = self.X_train.shape[0]
        utilities = np.zeros((2**n_players,))

        y_train_sorted = self.y_train[sortperm]
        for coalition_generator in product([False, True], repeat=self.X_train.shape[0]):
            coalition = np.array(list(coalition_generator))

            # Utility function according to equation (15) in Wang et al. (2024)
            y_val_mask = y_train_sorted == y_val
            y_other_mask = y_train_sorted == y_other
            # Mask of k nearest training points with class y_val or y_other
            k_nearest_with_relevant_class = _first_n_true(
                coalition & (y_val_mask | y_other_mask), self.k
            )
            y_val_nearest = y_val_mask & k_nearest_with_relevant_class
            y_other_nearest = y_other_mask & k_nearest_with_relevant_class
            utility = int(np.sum(weights[y_val_nearest]) >= np.sum(weights[y_other_nearest]))

            idx = np.sum(2 ** (sortperm[coalition]))
            utilities[idx] = utility

        game = ArrayGame(n_players=n_players, utility=utilities)
        iv = game.exact_values("SII", order=1)

        sv = np.zeros((n_players,), dtype=np.floating)
        for i in range(n_players):
            sv[i] = iv.values[iv.interaction_lookup[(i,)]]

        return sv

    def _get_training_data_weights_sorted(
        self, x_val: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Calculate distances and weights of all training data points (called X hereinafter) with respect to to a given validation data point x_val.

        Args:
            x_val: The validation data point.

        Returns:
            A tuple `(sortperm, dists, weights)`, where all are `numpy.ndarray`s with dimensions `(n_training_samples,)` and
                - `sortperm` is a permutation that sorts X by distance to x_val
                - `dists` contains the distance of each point in X to x_val
                - `weights` contains the corresponding weights for each point in X
        """
        distances, sortperm = self.model.kneighbors(
            x_val.reshape(1, -1), n_neighbors=self.X_train.shape[0]
        )
        return sortperm[0], sklearn_get_weights(distances, self.model.weights)[0]


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
