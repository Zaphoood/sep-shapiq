"""Tests the WKNN Explainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn import interaction_values_to_array
from shapiq_student.explainer.knn.weighted_knn import BruteForceWKNNExplainer, WeightedKNNExplainer


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


class TestWKNNValues:
    """Tests that the values calculated by WeightedKNNExplainer are (approximately) equal to the baseline brute-force implementation."""

    def test_wknn_exact_binary(self) -> None:
        """Tests that the results of WeightedKNNExplainer agree with baseline calculated using the same (discretized) weights for binary classification."""
        n_test_cases = 10
        n_train_min = 5
        n_train_max = 10
        n_bits = 4

        rng = np.random.default_rng(seed=43)

        for dataset, k in random_test_datasets(
            rng, n_test_cases, n_train_min=n_train_min, n_train_max=n_train_max, k_min=3, k_max=5
        ):
            self._compare_wknn_exact(WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits))

    def test_wknn_exact_multiclass(self) -> None:
        """Tests that the results of WeightedKNNExplainer agree with baseline calculated using the same (discretized) weights for multi-class classification."""
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
            self._compare_wknn_exact(WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits))

    def _compare_wknn_exact(self, test_case: WKNNTestCase) -> None:
        model = self._get_fitted_wknn_model(test_case)

        for class_index in range(len(set(test_case.y_train))):
            explainer_wang = WeightedKNNExplainer(
                model, class_index=class_index, n_bits=test_case.n_bits
            )
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

    def test_wknn_approximate(self) -> None:
        """Tests that the results of WeightedKNNExplainer with discretized weights are approximately equal to the baseline computed with continuous weights."""
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
            self._compare_wknn_approximate(
                WKNNTestCase.from_dataset(dataset, k=k, n_bits=n_bits), tolerance=tolerance
            )

    def _compare_wknn_approximate(self, test_case: WKNNTestCase, tolerance: float) -> None:
        model = self._get_fitted_wknn_model(test_case)

        total_error = 0
        for class_index in range(len(set(test_case.y_train))):
            # WeightedKNNExplainer will use discretized weights
            explainer = WeightedKNNExplainer(
                model, class_index=class_index, n_bits=test_case.n_bits
            )
            iv = explainer.explain(test_case.x_val)
            sv = interaction_values_to_array(iv)

            # BruteForceWKNNExplainer will use continuous weights
            explainer_brute = BruteForceWKNNExplainer(model, class_index=class_index)
            iv_brute = explainer_brute.explain_function(test_case.x_val)
            sv_brute = interaction_values_to_array(iv_brute)

            total_error += np.sum(np.abs(sv_brute - sv))

        assert total_error < tolerance

    def _get_fitted_wknn_model(self, test_case: WKNNTestCase) -> KNeighborsClassifier:
        model = KNeighborsClassifier(n_neighbors=test_case.k, weights="distance")
        model.fit(test_case.X_train, test_case.y_train)
        return model


class TestWKNNSanity:
    """Performs various sanity checks and tests helper functions."""

    def test_raises_unfitted_inadequate_model(self):
        """Tests that instantiating WeightedKNNExplainer with an unfitted model raises an exception."""
        model = KNeighborsClassifier(n_neighbors=3, weights="distance")

        with pytest.raises(NotFittedError):
            WeightedKNNExplainer(model, class_index=0)

    def test_raises_on_inadequate_model(self):
        """Tests that instantiating WeightedKNNExplainer with a model that uses unweighted weights raises an exception."""
        model = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        model.fit(np.array([[0]]), np.array([0]))

        with pytest.raises(ValueError, match="weights"):
            WeightedKNNExplainer(model, class_index=0)

    def test_raises_negative_discretization_bits(self):
        """Tests that instantiating WeightedKNNExplainer with a value of n_bits below zero."""
        model = KNeighborsClassifier(n_neighbors=3, weights="distance")
        model.fit(np.array([[0]]), np.array([0]))

        with pytest.raises(ValueError, match="bits"):
            WeightedKNNExplainer(model, class_index=0, n_bits=-1)

    def test_raises_invalid_k(self):
        """Tests that instantiating WeightedKNNExplainer with a value of one for parameter k."""
        model = KNeighborsClassifier(n_neighbors=1, weights="distance")
        model.fit(np.array([[0]]), np.array([0]))

        with pytest.raises(ValueError, match=r"value.*\bk\b"):
            WeightedKNNExplainer(model, class_index=0)

    def test_single_class(self):
        """Tests that if the training data only consists of a single class, all Shapley Values are zero."""
        n = 10
        rng = np.random.default_rng(seed=42)
        X_train = rng.normal(size=(n, 2))
        y_train = np.full((n,), "foo")
        x_val = rng.normal(size=(1, 2)).flatten()

        model = KNeighborsClassifier(n_neighbors=3, weights="distance")
        model.fit(X_train, y_train)

        explainer = WeightedKNNExplainer(model, class_index=0)

        sv = interaction_values_to_array(explainer.explain(x_val))

        assert np.allclose(sv, 0)

    def test_wknn_discretize_weights(self):
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

        explainer = WeightedKNNExplainer(model, class_index=class_index, n_bits=n_bits)
        sortperm, weights_prepared_sorted = explainer._get_prepared_weights(x_val)

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
