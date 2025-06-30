"""Implementation of the explainer for weighted KNN models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast
from typing_extensions import override

from .base import KNNExplainerBase, interaction_values_from_array

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
        indices = cast("npt.NDArray[np.int64]", np.sum(bases, axis=1))

        return self.utility[indices]


class WKNNExplainerBase(ABC, KNNExplainerBase):
    """Base class for WKNN explainers that provides a utility function for calculating weights of training data points."""

    def _get_training_data_dists_and_weights(
        self, x_val: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Calculate weights and sorting permutation training data points (called X_train hereinafter) with respect to to a given validation data point x_val.

        Args:
            x_val: The validation data point.

        Returns:
            A tuple `(sortperm, dists, weights)`, where all are `numpy.ndarray`s with dimensions `(n_training_samples,)` and
                - `sortperm` is a permutation that sorts X_train by distance to x_val
                - `weights` contains the corresponding weights for each point in X_train
        """
        distances, sortperm = cast(
            "tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]",
            self.model.kneighbors(x_val.reshape(1, -1), n_neighbors=self.X_train.shape[0]),
        )
        weights = 1 / distances
        return sortperm[0], weights[0]

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        sortperm, weights = self._get_training_data_weights_sorted(x)
        n_players = self.X_train.shape[0]
        sv = np.zeros((n_players,))
        n_classes = len(self.y_train_classes)

        if n_classes == 1:
            return interaction_values_from_array(np.zeros((n_players,), dtype=np.floating))

        for other_class in self.y_train_classes:
            if other_class == self.class_index:
                continue
            sv_current = self._explain_binary(self.class_index, other_class, sortperm, weights)
            sv += sv_current

        sv /= n_classes - 1

        return interaction_values_from_array(sv)

    @abstractmethod
    def _explain_binary(
        self,
        y_val: int,
        y_other: int,
        sortperm: npt.NDArray[np.integer],
        weights: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Computes the Shapley Values for a single binary-class classification game.

        Class ``y_val`` is the class to explain and ``y_other`` the other class in the binary classification setting. All other weights shall be ignored.
        """
        msg = "Method _explain_binary() must be implemented by each subclass."
        raise NotImplementedError(msg)

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


class BruteForceWKNNExplainer(WKNNExplainerBase):
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

    def _explain_binary(
        self,
        y_val: int,
        y_other: int,
        sortperm: npt.NDArray[np.integer],
        weights: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Computes the Shapley Values for a single binary-class classification game.

        Class ``y_val`` is the class to explain and ``y_other`` the other class in the binary classification setting. All other weights are ignored.
        """
        n_players = self.X_train.shape[0]
        utilities = np.zeros((2**n_players,))

        y_train_sorted = self.y_train[sortperm]
        y_val_mask = y_train_sorted == y_val
        y_other_mask = y_train_sorted == y_other
        for coalition_generator in product([False, True], repeat=self.X_train.shape[0]):
            coalition = np.array(list(coalition_generator))

            # Utility function according to equation (15) in Wang et al. (2024)

            # Mask of k nearest training points of current coalition with class y_val or y_other
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


class WKNNExplainer(WKNNExplainerBase):
    """Efficient implementation of WKNN according to `Wang et. al (2024)` [Wng24]_.

    References:
        .. [Wng24] Wang, Jiachen T., Prateek Mittal, and Ruoxi Jia. "Efficient data shapley for weighted nearest neighbor algorithms." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int,
        n_bits: int = 3,
    ) -> None:
        """Initializes the BruteForceWKNNExplainer.

        Args:
            model: The KNN model to explain.
            n_bits: The number of bits to use for the discretized weight space.
            class_index: The class index of the model to explain.
        """
        super().__init__(model, class_index)

        if self.k <= 1:
            msg = f"Only values of k > 1 are supported, but {self.k=}"
            raise ValueError(msg)

        if self.model.weights != "distance":
            msg = f"KNeighboursClassifier must use weights='distance', but has weights='{self.model.weights}'"
            raise ValueError(msg)

        self.n_bits = n_bits
        self.weights_space_size = self.k * 2**n_bits

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        # Number of training points
        n = len(self.y_train)
        n_classes = len(self.y_train_classes)

        # TODO(Zaphoood): Handle multi-class prediction
        if n_classes != 2:  # noqa: PLR2004
            msg = f"Multi-class prediction is not yet implemented (got {n_classes=})"
            raise NotImplementedError(msg)

        sortperm, weights = self._get_training_data_dists_and_weights(x)
        weights_signed_normalized = self._prepare_weights(weights, sortperm, self.class_index)
        weights_discrete = self._discretize_weights(weights_signed_normalized)

        sv = np.zeros(n)
        for i in range(n):
            fi = np.zeros((n, self.k - 1, self.weights_space_size))
            for m, w_m in enumerate(weights_discrete):
                if m == i:
                    continue
                fi[m, 1, w_m] = 1

        raise NotImplementedError(weights_discrete, sv)

    def _prepare_weights(
        self, weights: npt.NDArray[np.floating], sortperm: npt.NDArray[np.integer], y_pred: int
    ) -> npt.NDArray[np.floating]:
        """Normalize weights to interval [0, 1] and flip sign where y label disagrees with validation label.

        Args:
            weights: The original weights.
            sortperm: Sorting permutation of training data points according to weights.
            y_pred: The class predicted for the validation data point.

        Returns:
            An `np.ndarray` containing the prepared weights.
        """
        # Normalize weights to [0, 1]
        if np.max(weights) - np.min(weights) > 0:
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        # Change sign of weights where corresponding class disagrees with validation point prediction class
        weights[(self.y_train != y_pred)[sortperm]] *= -1

        return weights

    def _discretize_weights(self, weights: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Discretize weights according to the number of bits specified in the constructor.

        Args:
            weights: An `np.ndarray` containing weights to discretize.

        Returns:
            An `np.ndarray` of integers, which are indices into the discretized weight space ``W_(K)``.

        Examples:
            With `n_bits=3`, the input `[0.0, 0.1, 0.3, 0.8, 1.0]` will result in `[0, 1, 2, 6, 8]`.

        """
        return np.round(weights * 2**self.n_bits).astype(int)
