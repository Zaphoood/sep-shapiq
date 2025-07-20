"""Tests for the threshold nearest neighbor explainer."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
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
        class_index = 1
        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)

        assert np.allclose(tnn_explainer.X_train, X_train)
        assert np.allclose(tnn_explainer.y_train_classes[tnn_explainer.y_train_indices], y_train)
        assert tnn_explainer.tau == tau
        assert set(tnn_explainer.y_train_classes) == set(y_train)
        assert tnn_explainer.class_index == class_index

    def test_raises_unfitted_model(self):
        """Tests that instantiating ThresholdNNExplainer with an unfitted model raises an exception."""
        model = RadiusNeighborsClassifier(radius=1)

        with pytest.raises(NotFittedError):
            ThresholdNNExplainer(model, class_index=0)

    def test_zero_radius_model(self):
        """Tests behavior when calling RadiusNeighborsClassifier with radius zero. Should return all shapley values zero."""
        X_train = np.array([[1, 2, 3], [4, 5, 6]])
        y_train = np.array([0, 1])
        x_val = np.array([10, 10, 10])
        tau = 0
        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)
        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=1)
        sv = interaction_values_to_array(tnn_explainer.explain(x_val))

        assert np.allclose(sv, 0)

    def test_one_same_label_neighbor_in_threshold(self):
        """Tests ThresholdNNExplainer behavior when one neighbor is within threshold tau and of same label.

        Not all shapley values should be zero.

        The resulting Shapley values should be 0.5 for exactly one player and zero for the other. Reference: Formula (7) in Wang et al. [Wng23]_.
        """
        X_train = np.array([[1, 1, 1], [11, 11, 11]])
        y_train = np.array([0, 1])
        tau = 2
        x_val = np.array([2, 1, 1])
        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)
        class_index = 0
        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)
        sv = interaction_values_to_array(tnn_explainer.explain(x_val))

        assert np.sum(np.isclose(sv, 0, atol=1e-6)) == 1
        assert np.sum(np.isclose(sv, 0.5, atol=1e-6)) == 1

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

    def test_two_same_labels_have_same_sv(self):
        """Tests that two same label training points within threshold have the same Shapley value."""
        X_train = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 4], [5, 1, 6]])
        y_train = np.array([0, 0, 1, 1])
        tau = 10
        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)
        x_val = np.array([1, 1, 1])

        for i in range(len(set(y_train))):
            tnn_explainer = ThresholdNNExplainer(radius_model, class_index=i)
            tnn_sv = interaction_values_to_array(tnn_explainer.explain(x_val))
            assert np.isclose(tnn_sv[0], tnn_sv[1])
            assert np.isclose(tnn_sv[2], tnn_sv[3])

    def test_one_different_label_neighbor_in_threshold(self):
        """Tests ThresholdNNExplainer behavior when one neighbor is within threshold tau and of different label.

        Not all Shapley values should be zero.

        Result should be -0.5. Reference: Formula (7) in Wang et al.
        """
        X_train = np.array([[1, 1, 1], [11, 11, 11]])
        y_train = np.array([0, 1])
        tau = 2
        x_val = np.array([2, 1, 1])
        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)
        class_index = 1
        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)
        sv = interaction_values_to_array(tnn_explainer.explain(x_val))

        assert np.sum(np.isclose(sv, 0, atol=1e-6)) == 1
        assert np.sum(np.isclose(sv, -0.5, atol=1e-6)) == 1

    def test_point_on_threshold(self):
        """Tests that training point located exactly on threshold will be included and return non-zero shapley values."""
        test_cases = 6
        for i in (number + 1 for number in range(test_cases)):
            tau = i
            x_val = np.array([0, 0, 0])
            X_train = np.array([[tau, 0, 0], [0, tau, 0], [0, 0, tau]])
            y_train = [0, 0, 0]
            radius_model = RadiusNeighborsClassifier(radius=tau)
            radius_model.fit(X_train, y_train)
            class_index = 0
            tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)
            sv = interaction_values_to_array(tnn_explainer.explain(x_val))
            assert np.any(sv != 0)

    def test_point_slightly_outside_threshold(self):
        """Tests that training points slightly outside of the threshold will not be included and return zero Shapley values."""
        test_cases = 6
        for i in (number + 1 for number in range(test_cases)):
            tau = i
            offset = 0.1
            x_val = np.array([0, 0, 0])
            X_train = np.array([[tau + offset, 0, 0], [0, tau + offset, 0], [0, 0, tau + offset]])
            y_train = [0, 0, 0]
            radius_model = RadiusNeighborsClassifier(radius=tau)
            radius_model.fit(X_train, y_train)
            class_index = 0
            tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)
            sv = interaction_values_to_array(tnn_explainer.explain(x_val))
            assert np.allclose(sv, 0)

    def test_class_index_out_of_bounds(self):
        """Tests behavior when class_index is out of bounds.

        All Shapley values should have the same negative value.
        """
        X_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [11, 11, 11]])
        y_train = np.array([0, 1, 0, 1, 0, 1])
        tau = 20
        x_val = np.array([2, 1, 1])
        radius_model = RadiusNeighborsClassifier(radius=tau)
        radius_model.fit(X_train, y_train)
        class_index = 7
        tnn_explainer = ThresholdNNExplainer(radius_model, class_index=class_index)
        sv = interaction_values_to_array(tnn_explainer.explain(x_val))
        assert np.all(sv < 0)
        assert len(set(sv)) == 1

    def test_compare_with_brute_force_on_iris(self):
        """Tests the correctness of the ThresholdNNExplainer by comparing its results to the baseline brute force implementation.

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

    def test_mode(self):
        """Tests that the explainer mode is set correctly."""
        nn_model = RadiusNeighborsClassifier()
        nn_model.fit(np.array([[0]]), np.array([0]))

        explainer = ThresholdNNExplainer(nn_model, class_index=0)
        assert explainer.mode == "threshold"
