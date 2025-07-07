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

    @property
    def n_train(self) -> int:
        """The number of training points."""
        return self.X_train.shape[0]


def random_test_datasets(
    rng: np.random.Generator,
    n_test_cases: int,
    n_train_min: int,
    n_train_max: int,
    k_min: int,
    k_max: int,
    n_classes: int = 2,
) -> Iterator[tuple[Dataset, int]]:
    """Randomly generates binary datasets for testing."""
    for _ in range(n_test_cases):
        n_test_cases = int(rng.integers(n_train_min, n_train_max, endpoint=True))
        X_train = rng.normal(size=(n_test_cases, 2))
        y_train = rng.integers(0, n_classes, size=n_test_cases)
        x_val = rng.normal(size=(1, 2))[0]

        k = int(rng.integers(k_min, k_max, endpoint=True))

        yield Dataset(X_train=X_train, y_train=y_train, x_val=x_val), k


def test_wknn_exact_binary() -> None:
    """Tests that the results of WKNNExplainer agree with those of BruteForceWKNNExplainer for binary classification when using the same (discretized) weights."""
    n_test_cases = 10
    n_train_min = 5
    n_train_max = 10
    n_bits = 4

    rng = np.random.default_rng(seed=43)

    for dataset, k in random_test_datasets(
        rng, n_test_cases, n_train_min=n_train_min, n_train_max=n_train_max, k_min=3, k_max=5
    ):
        _compare_wknn_exact(WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits))


def test_wknn_exact_multiclass() -> None:
    """Tests that the results of WKNNExplainer agree with those of BruteForceWKNNExplainer for multi-class classification when using the same (discretized) weights."""
    n_test_cases = 10
    n_train_min = 9
    n_train_max = 10
    n_bits = 4

    rng = np.random.default_rng(seed=43)

    for n_classes in range(3, 6):
        dataset, k = next(
            random_test_datasets(
                rng,
                n_test_cases,
                n_train_min=n_train_min,
                n_train_max=n_train_max,
                k_min=3,
                k_max=5,
                n_classes=n_classes,
            )
        )
        _compare_wknn_exact(WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits))


def _get_fitted_wknn_model(test_case: WKNNTestCase) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=test_case.k, weights="distance")
    model.fit(test_case.X_train, test_case.y_train)
    return model


def _compare_wknn_exact(test_case: WKNNTestCase) -> None:
    model = _get_fitted_wknn_model(test_case)

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


def test_wknn_approximate() -> None:
    """Tests that the Shapley Values computed by WKNNExplainer with discretized weights are approximately equal to the values computed by BruteForceWKNNExplainer with continuous weights."""
    n_test_cases = 3
    n_train_min = 10
    n_train_max = 10
    n_bits = 10
    tolerance = 1e-10

    rng = np.random.default_rng(seed=43)

    for dataset, k in random_test_datasets(
        rng,
        n_test_cases,
        n_train_min=n_train_min,
        n_train_max=n_train_max,
        k_min=3,
        k_max=5,
    ):
        _compare_wknn_approximate(
            WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits), tolerance=tolerance
        )


def _compare_wknn_approximate(test_case: WKNNTestCase, tolerance: float) -> None:
    model = _get_fitted_wknn_model(test_case)

    total_error = 0
    for class_index in range(len(set(test_case.y_train))):
        # WKNNExplainer will use discretized weights
        explainer = WKNNExplainer(model, class_index=class_index, n_bits=test_case.n_bits)
        iv = explainer.explain(test_case.x_val)
        sv = interaction_values_to_array(iv)

        # BruteForceWKNNExplainer will use continuous weights
        explainer_brute = BruteForceWKNNExplainer(model, class_index=class_index)
        iv_brute = explainer_brute.explain_function(test_case.x_val)
        sv_brute = interaction_values_to_array(iv_brute)

        total_error += np.sum(np.abs(sv_brute - sv))

    assert total_error < tolerance


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
    test_wknn_exact_multiclass()
