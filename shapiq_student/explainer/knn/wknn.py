"""Implementation of the explainer for weighted KNN models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast, overload
from typing_extensions import override

from .base import KNNExplainerBase, interaction_values_from_array

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq.interaction_values import InteractionValues
    from sklearn.neighbors import KNeighborsClassifier

from itertools import product

import numpy as np
from scipy.special import comb
from shapiq.games import Game


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

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n_players = self.X_train.shape[0]

        n_classes = len(self.y_train_classes)
        if n_classes == 1:
            return interaction_values_from_array(np.zeros((n_players,), dtype=np.float64))

        sortperm, weights = self._get_normalized_weights(x)

        sv = np.zeros((n_players,))
        for other_class_index in range(n_classes):
            if other_class_index == self.class_index:
                continue
            sv_current = self._explain_binary(
                self.class_index, other_class_index, sortperm, weights
            )
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

        Only training data points which have class index ``y_val`` or ``y_other`` shall be considered and all others ignored.

        Args:
            y_val: The index of the class to explain.
            y_other: The index of the other class to consider for the binary sub-game.
            sortperm: Sorting permutation of the training data points with respect to weights.
            weights: Array of weights assigned to each training data point.
        """
        msg = "Method _explain_binary() must be implemented by each subclass."
        raise NotImplementedError(msg)

    def _get_normalized_weights(
        self, x_val: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Calculate normalized weights of training data points with respect to a validation data point.

        Args:
            x_val: The validation data point.

        Returns:
            A tuple ``(sortperm, weights)``, where both are of type ``numpy.ndarray`` with dimensions ``(n_training_samples,)`` and
                - ``sortperm`` is a permutation that sorts the training data points by decreasing weight
                - ``weights`` contains the weights for each training data point, normalized to the interval [0, 1]
        """
        distances, sortperm = self.model.kneighbors(
            x_val.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=True
        )
        distances = distances[0]
        sortperm = sortperm[0]

        weights = (distances[-1] - distances) / (distances[-1] - distances[0])

        return sortperm, weights


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
        n_players = self.X_train.shape[0]
        utilities = np.zeros((2**n_players,))

        y_train_sorted = self.y_train_indices[sortperm]
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

        sv = np.zeros((n_players,), dtype=np.float64)
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
        """Initializes the WKNNExplainer.

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
        self.weights_space_size = 2 * self.k * 2**n_bits + 1
        # Index at which weight 0.0 is mapped to in the discrete weight space
        self.weights_space_zero = self.k * 2**n_bits

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        # TODO(Zaphoood): Consider removing aliases
        # Convenience aliases
        x_val = x
        y_val = self.class_index

        # Number of training points
        n = len(self.y_train_indices)
        n_classes = len(self.y_train_classes)

        # TODO(Zaphoood): Handle multi-class prediction
        if n_classes != 2:  # noqa: PLR2004
            msg = f"Multi-class prediction is not yet implemented (got {n_classes=})"
            raise NotImplementedError(msg)

        sortperm, weights = self._get_normalized_weights(x_val)
        # Change sign of weights where class disagrees with class of validation point
        weights[(self.y_train_indices != self.class_index)[sortperm]] *= -1
        weights_discrete = self._discretize_weight(weights)

        sv = np.zeros(n)

        for i in range(n):
            y_i = cast("int", self.y_train_indices[sortperm[i]])
            f_i = self._compute_f_i(i, n, weights_discrete)
            r_i = self._compute_r_i(i, n, f_i, y_i, y_val, weights_discrete)
            g_i = self._compute_g_i(i, n, f_i, y_i, y_val, weights_discrete)

            sv[sortperm[i]] = self._compute_single_shapley_value(i, n, r_i, g_i, weights_discrete)

        return interaction_values_from_array(sv)

    def _compute_f_i(
        self,
        i: int,
        n: int,
        weights_discrete: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        f_i = np.zeros((n, self.k - 1, self.weights_space_size))

        for m, weight_m in enumerate(weights_discrete):
            if m == i:
                continue
            f_i[m, 0, weight_m] = 1

        for l in range(1, self.k - 1):  # noqa: E741
            for m in range(l, n):
                if m == i:
                    continue
                weight_m = weights_discrete[m]
                for s in range(self.weights_space_size):
                    for t in range(m - 1):
                        s_minus_weight_m = self._discrete_weight_sub(s, weight_m)
                        if 0 <= s_minus_weight_m < self.weights_space_size:
                            f_i[m, l, s] += f_i[t, l - 1, s_minus_weight_m]

        return f_i

    def _compute_r_i(
        self,
        i: int,
        n: int,
        f_i: npt.NDArray[np.floating],
        y_i: int,
        y_val: int,
        weights_discrete: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        r_i = np.zeros((n,))
        for m in range(max(i + 1, self.k), n):
            if y_i == y_val:
                weight_range_begin = self._flip_weight_sign(weights_discrete[i])
                weight_range_end = self._flip_weight_sign(weights_discrete[m])
            else:
                weight_range_begin = self._flip_weight_sign(weights_discrete[m])
                weight_range_end = self._flip_weight_sign(weights_discrete[i])

            if weight_range_begin < weight_range_end:
                for t in range(m - 1):
                    for s in range(weight_range_begin, weight_range_end):
                        r_i[m] += f_i[t, self.k - 2, s]

        return r_i

    def _compute_g_i(
        self,
        i: int,
        n: int,
        f_i: npt.NDArray[np.floating],
        y_i: int,
        y_val: int,
        weights_discrete: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        g_i = np.zeros((self.k,))
        g_i[0] = 1 if self._is_weight_negative(weights_discrete[i]) else 0
        for l in range(1, self.k):  # noqa: E741
            if y_i == y_val:
                weight_range_begin = self._flip_weight_sign(weights_discrete[i])
                weight_range_end = self.weights_space_zero
            else:
                weight_range_begin = self.weights_space_zero
                weight_range_end = self._flip_weight_sign(weights_discrete[i])

            if weight_range_begin < weight_range_end:
                for m in range(n):
                    if m == i:
                        continue
                    for s in range(weight_range_begin, weight_range_end):
                        g_i[l] += f_i[m, l - 1, s]

        return g_i

    def _compute_single_shapley_value(
        self,
        i: int,
        n: int,
        r_i: npt.NDArray[np.floating],
        g_i: npt.NDArray[np.floating],
        weights_discrete: npt.NDArray[np.integer],
    ) -> float:
        weight_sign = self._weight_sign(weights_discrete[i])
        first_summand = sum(g_i[l] / comb(n - 1, l) for l in range(self.k)) / n  # noqa: E741
        second_summand = sum(
            r_i[m - 1] / (m * comb(m - 1, self.k)) for m in range(max(i + 2, self.k + 1), n + 1)
        )

        return weight_sign * (first_summand + second_summand)

    @override
    def _explain_binary(
        self,
        y_val: int,
        y_other: int,
        sortperm: npt.NDArray[np.integer],
        weights: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        msg = f"{self.__class__.__name__} does not implement _explain_binary() yet"
        raise NotImplementedError(msg)

    @overload
    def _discretize_weight(self, weight: float) -> int: ...

    @overload
    def _discretize_weight(self, weight: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]: ...

    def _discretize_weight(
        self, weight: float | npt.NDArray[np.floating]
    ) -> int | npt.NDArray[np.integer]:
        """Turns floating-point weight into an integer index in the discretized weights space.

        Weight ``-k`` will be mapped to index ``0`` and weight ``k`` to index ``2 * k * 2**n_bits``.
        """
        return self.weights_space_zero + np.round(weight * 2**self.n_bits).astype(int)

    @overload
    def _undiscretize_weight(self, weight_discrete: int) -> float: ...

    @overload
    def _undiscretize_weight(
        self, weight_discrete: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]: ...

    def _undiscretize_weight(
        self, weight_discrete: int | npt.NDArray[np.integer]
    ) -> float | npt.NDArray[np.floating]:
        """Turns discrete weight index into the corresponding floating point weight."""
        return (weight_discrete - self.weights_space_zero) / (2**self.n_bits)

    def _discrete_weight_sub(self, a_discrete: int, b_discrete: int) -> int:
        """Computes ``a - b`` for two discrete weight indices."""
        return self.weights_space_zero + a_discrete - b_discrete

    @overload
    def _flip_weight_sign(self, weight_discrete: int) -> int: ...

    @overload
    def _flip_weight_sign(
        self, weight_discrete: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]: ...

    def _flip_weight_sign(
        self, weight_discrete: int | npt.NDArray[np.integer]
    ) -> int | npt.NDArray[np.integer]:
        """Given a discretized weight index, returns the discretized index of the corresponding weight with the sign flipped."""
        return 2 * self.weights_space_zero - weight_discrete

    @overload
    def _is_weight_negative(self, weight_discrete: int) -> int: ...

    @overload
    def _is_weight_negative(
        self, weight_discrete: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]: ...

    def _is_weight_negative(
        self, weight_discrete: int | npt.NDArray[np.integer]
    ) -> int | npt.NDArray[np.integer]:
        """Checks whether the weight corresponding to a discretized weight index is negative."""
        return weight_discrete < self.weights_space_zero

    def _weight_sign(self, weight_discrete: int) -> int:
        """Implements the sign function for discretized weights.

        Given some discretized weight index ``weight_discrete`` with corresponding weight ``w``, returns 1 if ``w > 0``, -1 if ``w < 0``, and 0 if ``w == 0```
        """
        if weight_discrete > self.weights_space_zero:
            return 1
        if weight_discrete < self.weights_space_zero:
            return -1

        return 0
