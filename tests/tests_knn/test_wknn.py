"""Tests the WKNN Explainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn.wknn import BruteForceWKNNExplainer, WKNNExplainer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


@dataclass
class WKNNTestCase:
    """Defines a test case for a WKNN classifier."""

    X_train: npt.NDArray[np.floating]
    y_train: npt.NDArray[np.object_ | np.integer]
    x_val: npt.NDArray[np.floating]
    k: int
    n_bits: int


def test_wknn_hardcoded_example():
    """Tests that the Shapley Values computed by WKNNExplainer are correct for a hard-coded example."""
    test_cases = [
        WKNNTestCase(
            X_train=np.array([[-8, 0], [-1.5, 0], [-0.5, 0], [0.5, 0], [1.5, 0]]),
            y_train=np.array(["foo", "foo", "foo", "bar", "bar"]),
            x_val=np.array([[0.2, 0]]),
            k=3,
            n_bits=4,
        ),
        WKNNTestCase(
            X_train=np.array(
                [
                    [0.5, -0.14],
                    [0.65, 1.52],
                    [-0.23, -0.23],
                    [1.58, 0.77],
                    [-0.47, 0.54],
                    [-0.46, -0.47],
                    [0.24, -1.91],
                    [-1.72, -0.56],
                    [-1.01, 0.31],
                    [-0.91, -1.41],
                ]
            ),
            y_train=np.array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0]),
            x_val=np.array([0, 0]),
            k=3,
            n_bits=3,
        ),
    ]

    for test_case in test_cases:
        _check_wknn_test_case(test_case)


def _check_wknn_test_case(test_case: WKNNTestCase) -> None:
    n = test_case.X_train.shape[0]
    model = KNeighborsClassifier(n_neighbors=test_case.k, weights="distance")
    model.fit(test_case.X_train, test_case.y_train)

    for class_index in range(len(set(test_case.y_train))):
        explainer_wang = WKNNExplainer(model, class_index=class_index, n_bits=test_case.n_bits)
        iv_wang = explainer_wang.explain(test_case.x_val)
        sv_wang = _interaction_values_to_array(iv_wang)

        explainer_brute = BruteForceWKNNExplainer(
            model,
            class_index=class_index,
            n_bits=test_case.n_bits,
        )
        iv_brute = explainer_brute.explain_function(test_case.x_val)
        sv_brute = _interaction_values_to_array(iv_brute)

        print(f"{class_index=}")
        print(f"wang\t{np.round(sv_wang, 3)}\t{np.round(np.sum(sv_wang), 10)}")
        print(f"brute\t{np.round(sv_brute, 3)}\t{np.round(np.sum(sv_brute), 10)}")

        # Test that SV values agree qualitatively with brute-force calculation, meaning that weak inequalities are equivalent
        for i in range(n):
            for j in range(i, n):
                # According to Appendix E, we have v_i >= v_j implies v_i^disc >= v_j^disc
                assert not (sv_brute[i] >= sv_brute[j]) or (sv_wang[i] >= sv_wang[j])


def test_wknn_random() -> None:
    """Tests that the results of WKNNExplainer agree with those of BruteForceWKNNExplainer using randomly generated test cases."""
    n_test_cases = 3
    min_training_points = 5
    max_training_points = 10
    n_bits = 4
    k = 3

    rng = np.random.default_rng(seed=43)

    for _ in range(n_test_cases):
        n = int(rng.integers(min_training_points, max_training_points))
        X_train, y_train = _generate_binary_split_training_data(rng, n)
        x_val = rng.normal(size=(1, 2))[0]

        _check_wknn_test_case(WKNNTestCase(X_train, y_train, x_val, k=k, n_bits=n_bits))


def _generate_binary_split_training_data(
    rng: np.random.Generator,
    n: int,
    split_vec: npt.NDArray[np.floating] | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int64]]:
    if split_vec is None:
        split_vec = np.array([1, 1])

    X = rng.normal(size=(n, 2))
    split_vec = np.array([1, 1])
    y = np.zeros((n,), dtype=np.int64)
    y[X.dot(split_vec) > 0] = 1

    return X, y


def test_wknn_discretize_weights():
    """Tests the pre-processing of weights involved in the WKNN algorithm, and the weight sign flipping method."""
    # Distances are [1, 0, 1, 4, 4] -> normalized weights are [3/4, 4/4, 3/4, 0, 0]
    X_train = np.array([[-1, 0], [0, 0], [1, 0], [4, 0]])
    y_train = np.array([0, 1, 1, 0])
    x_val = np.array([0, 0])
    class_index = 1
    k = 3
    n_bits = 2

    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    model.fit(X_train, y_train)

    explainer = WKNNExplainer(model, class_index=class_index, n_bits=n_bits)
    sortperm, weights_prepared_sorted = explainer._get_discrete_weights(x_val)

    print(f"{weights_prepared_sorted=}")
    weights_prepared = np.zeros_like(sortperm)
    weights_prepared[sortperm] = weights_prepared_sorted
    print(f"{weights_prepared=}")

    assert explainer.weights_space_size == 2 * k * 2**n_bits + 1
    assert weights_prepared.dtype in (np.int64, np.int32)

    zero_idx = k * 2**n_bits
    assert weights_prepared[0] == zero_idx - 3
    assert weights_prepared[1] == zero_idx + 4
    assert weights_prepared[2] == zero_idx + 3

    assert weights_prepared[3] == zero_idx

    assert np.all(
        explainer._flip_weight_sign(explainer._flip_weight_sign(weights_prepared))
        == weights_prepared
    )

    assert explainer._flip_weight_sign(weights_prepared[0]) == weights_prepared[2]
    assert explainer._flip_weight_sign(weights_prepared[2]) == weights_prepared[0]


def _interaction_values_to_array(shapley_values: InteractionValues) -> npt.NDArray[np.floating]:
    """Extracts Shapley Values from an ``InteractionValues`` object and returns them in an array, ordered by player index.

    Args:
        shapley_values: An InteractionValues object with ``max_order==1``

    Returns:
        An ``np.ndarray`` of shape (n_players,) containing at index i the Shapley value of player i.
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


if __name__ == "__main__":
    test_wknn_random()
