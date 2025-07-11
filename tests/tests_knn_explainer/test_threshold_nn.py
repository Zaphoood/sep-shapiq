"""Tests for the threshold nearest-neighbor explainer."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier

from shapiq_student.explainer.knn.base import interaction_values_to_array
from shapiq_student.explainer.knn.threshold_nn import ThresholdNNExplainer, _BruteForceTNNExplainer


class TestThresholdNNExplainer:
    """Tests for the ThresholdNNExplainer class."""

    def test_extract_parameters_from_radius_model(self):
        """Tests parameters are successfully extracted from RadiusNeighborsClassifier model."""
        X_train = np.array([[1, 2, 3], [1, 2, 3]])
        y_train = np.array([0, 1])
        tau = 1

        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)

        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=1)

        assert np.allclose(tnn_explainer.X_train, X_train)
        assert np.allclose(tnn_explainer.y_train, y_train)
        assert tnn_explainer.tau == tau

    def test_no_neighbors_in_threshold(self):
        """Tests behavior when no neighbors are within threshold tau. All shapley values should be zero."""
        X_train = np.array([[10, 10, 10], [11, 11, 11]])
        y_train = np.array([0, 1])
        tau = 1
        x_val = np.array([1, 1, 1])
        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)
        class_index = 0
        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)
        sv_array_tnn = interaction_values_to_array(tnn_explainer.explain(x_val))

        assert np.allclose(sv_array_tnn, 0)

    def test_compare_with_brute_force_on_iris(self):
        """Tests the correctness of the TNN explainer by comparing its results to the baseline brute force implementation.

        The model used for testing is trained on a small part of the "Iris" dataset.
        """
        iris = load_iris()
        X = cast("npt.NDArray[np.floating]", iris.data)
        y = cast("npt.NDArray[np.floating]", iris.target)

        # Limit the training data set because the brute force algorithm is really slow
        n_train_max = 12
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=42)
        X_train = X_train[:n_train_max]
        y_train = y_train[:n_train_max]
        print(f"{X_train.shape=}")

        x_val = X_test[0]
        n_classes = len(set(y))

        tau = 1
        model = RadiusNeighborsClassifier(radius=tau)
        model.fit(X_train, y_train)

        for class_index in range(n_classes):
            brute_explainer = _BruteForceTNNExplainer(model, class_index=class_index)
            brute_iv = interaction_values_to_array(brute_explainer.explain(x_val))
            tnn_explainer = ThresholdNNExplainer(model, class_index=class_index)
            tnn_iv = interaction_values_to_array(tnn_explainer.explain(x_val))

            assert np.allclose(brute_iv, tnn_iv)
