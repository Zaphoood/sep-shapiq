"""Implementation of the explainer for weighted KNN models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

from .base import KNNExplainerBase

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
    ) -> None:
        """Initializes the BruteForceWKNNExplainer.

        Args:
            model: The KNN model to explain.
        """
        super().__init__(model)

        if self.model.weights != "distance":
            msg = f"KNeighboursClassifier must use weights='distance', but has weights='{self.model.weights}'"
            raise ValueError(msg)

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n_players = self.X_train.shape[0]
        y_pred = self.model.predict(x.reshape(1, -1))[0]

        sortperm, weights = self._get_training_data_weights_sorted(x)

        utility = np.zeros((2**n_players,))

        indices = cast("npt.NDArray[np.integer]", np.arange(n_players))
        for mask_generator in product([False, True], repeat=self.X_train.shape[0]):
            mask = np.array(list(mask_generator))
            k_nearest_indices = indices[mask][: self.k]

            # Maximum score of any class
            score_max = float("-inf")
            # Score of the class that was originally predicted
            score_y_pred: float | None = None

            for class_ in self.classes:
                nearest_with_class = k_nearest_indices[
                    self.y_train[sortperm[k_nearest_indices]] == class_
                ]

                score = np.sum(weights[nearest_with_class], dtype=float)
                score_max = max(score_max, score)
                if class_ == y_pred:
                    score_y_pred = score

            if score_y_pred is None:
                msg = "Unreachable. y_pred not in self.classes"
                raise RuntimeError(msg)

            idx = np.sum(2 ** (sortperm[mask]))
            utility[idx] = int(score_y_pred == score_max)

        game = ArrayGame(n_players=n_players, utility=utility)

        return game.exact_values("SII", order=1)

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
