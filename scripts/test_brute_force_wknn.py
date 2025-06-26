from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn.wknn import BruteForceWKNNExplainer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


def _get_ordered_values(shapley_values: InteractionValues) -> npt.NDArray[np.floating]:
    """Return shapley values ordered by player index.

    Args:
        shapley_values: An InteractionValues object with max_order==1

    Returns:
        An np.ndarray of shape (n_players,) containing at index i the Shapley value of player i.
    """
    if shapley_values.max_order != 1:
        msg = f"Max order must be 1 but was {shapley_values.max_order}"
        raise ValueError(msg)

    out = np.zeros((shapley_values.n_players,))

    for coalition, lookup_idx in shapley_values.interaction_lookup.items():
        if coalition == ():
            continue
        out[coalition[0]] = shapley_values.values[lookup_idx]

    return out


def test_random(n: int = 5, k: int = 3):
    X_train, y_train = generate_binary_split_training_data(n=n)
    x_val = np.random.randn(1, 2).flatten()

    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X_train, y_train)

    explain_print(model, x_val)


def generate_binary_split_training_data(
    n: int, split_vec=np.array([1, 1])
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
    X = np.random.randn(n, 2)
    split_vec = np.array([1, 1])
    y = np.empty((n,), dtype=object)
    y[X.dot(split_vec) > 0] = "foo"
    y[X.dot(split_vec) <= 0] = "bar"

    return X, y


def test_handpicked():
    X_train = np.array([[-8, 0], [-1.5, 0], [-0.5, 0], [0.5, 0], [1.5, 0]])
    y_train = np.array([0, 0, 0, 1, 1])
    x_val = np.array([[0.2, 0]])
    k = 3

    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X_train, y_train)

    explain_print(model, x_val)


def explain_print(
    model: KNeighborsClassifier,
    x_val: npt.NDArray[np.floating],
) -> None:
    brute_explainer = BruteForceWKNNExplainer(model)
    brute_shapley_values = brute_explainer.explain(x_val)
    print(_get_ordered_values(brute_shapley_values))


def main():
    # test_random(n=14)
    test_handpicked()


if __name__ == "__main__":
    main()
