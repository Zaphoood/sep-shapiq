"""Tests the WKNN Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn.base import interaction_values_to_array
from shapiq_student.explainer.knn.wknn import BruteForceWKNNExplainer, WKNNExplainer


@dataclass
class Dataset:
    """Dataset for unit tests that groups X_train, y_train and a validation point x_val."""

    X_train: npt.NDArray[np.floating]
    y_train: npt.NDArray[np.int64]
    x_val: npt.NDArray


@dataclass
class WKNNTestCase:
    """Defines a test case for a WKNN classifier."""

    X_train: npt.NDArray[np.floating]
    y_train: npt.NDArray[np.object_ | np.integer]
    x_val: npt.NDArray[np.floating]
    k: int
    n_bits: int

    @classmethod
    def from_dataset(cls, dataset: Dataset, k: int, n_bits: int) -> WKNNTestCase:
        """Converts a Dataset to a WKNNTestCase."""
        return cls(dataset.X_train, dataset.y_train, dataset.x_val, k=k, n_bits=n_bits)


def test_wknn_sv_agrees_with_brute_force() -> None:
    """Tests that the results of WKNNExplainer agree with those of BruteForceWKNNExplainer using randomly generated test cases."""
    n_test_cases = 10
    n_train_min = 5
    n_train_max = 10
    n_bits = 4

    rng = np.random.default_rng(seed=43)

    for dataset, k in random_binary_test_datasets(
        rng, n_test_cases, n_train_min=n_train_min, n_train_max=n_train_max, k_min=3, k_max=5
    ):
        _check_wknn_test_case_with_brute_froce(
            WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits)
        )


def random_binary_test_datasets(
    rng: np.random.Generator,
    n: int,
    n_train_min: int,
    n_train_max: int,
    k_min: int,
    k_max: int,
) -> Iterator[tuple[Dataset, int]]:
    """Randomly generates binary datasets for testing."""
    for _ in range(n):
        n = int(rng.integers(n_train_min, n_train_max, endpoint=True))
        X_train = rng.normal(size=(n, 2))
        y_train = rng.integers(0, 2, size=n)
        x_val = rng.normal(size=(1, 2))[0]

        k = int(rng.integers(k_min, k_max, endpoint=True))

        yield Dataset(X_train=X_train, y_train=y_train, x_val=x_val), k


def _check_wknn_test_case_with_brute_froce(test_case: WKNNTestCase) -> None:
    model = KNeighborsClassifier(n_neighbors=test_case.k, weights="distance")
    model.fit(test_case.X_train, test_case.y_train)

    for class_index in range(len(set(test_case.y_train))):
        explainer_wang = WKNNExplainer(model, class_index=class_index, n_bits=test_case.n_bits)
        iv_wang = explainer_wang.explain(test_case.x_val)
        sv_wang = interaction_values_to_array(iv_wang)

        explainer_brute = BruteForceWKNNExplainer(
            model,
            class_index=class_index,
            # Discretize weights in brute force explainer
            n_bits=test_case.n_bits,
        )
        iv_brute = explainer_brute.explain_function(test_case.x_val)
        sv_brute = interaction_values_to_array(iv_brute)

        assert np.allclose(sv_brute, sv_wang)


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

    weights_prepared = np.zeros_like(sortperm)
    weights_prepared[sortperm] = weights_prepared_sorted

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


if __name__ == "__main__":
    test_wknn_sv_agrees_with_brute_force()
