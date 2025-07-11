"""Test the functionality of NormalKNNExplainer."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from shapiq_student.explainer.knn import interaction_values_to_array
from shapiq_student.explainer.knn.normal_knn import (
    NormalKNNExplainer,
    _BruteForceNormalKNNExplainer,
)


class TestNormalKNNExplainer:
    """Tests for the NormalKNNExplainer class."""

    def test_single_same_label_training_point(self):
        """Tests Explainer behavior when model is trained on one single same label neighbor.

        Expected behavior: one shapley value with value 1.
        """
        X_train = np.array([[1, 1, 1]])
        y_train = [0]
        k = 1
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        x_val = np.array([2, 2, 2])
        class_index = 0
        explainer = NormalKNNExplainer(model, class_index=class_index)
        sv = interaction_values_to_array(explainer.explain(x_val))

        assert len(sv) == 1
        assert np.sum(sv) == 1

    def test_single_different_label_training_point(self):
        """Tests NormalKNNExplainer behavior when model is trained on one single neighbor of different label.

        Expected behavior: one shapley value with value 0.
        """
        X_train = np.array([[1, 1, 1]])
        y_train = [0]
        k = 1
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        x_val = np.array([2, 2, 2])
        class_index = 1
        explainer = NormalKNNExplainer(model, class_index=class_index)
        sv = interaction_values_to_array(explainer.explain(x_val))

        assert len(sv) == 1
        assert np.sum(sv) == 0

    def test_all_same_label_neighbors(self):
        """Tests NormalKNNExplainer behavior when a model is trained on only same label neighbors and k is equal to amount of training points.

        Expected behavior: All shapley values should have the same constant value 1/n.
        """
        X_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        y_train = [0, 0, 0, 0]
        k = len(X_train)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        x_val = np.array([2, 2, 2])
        class_index = 0
        explainer = NormalKNNExplainer(model, class_index=class_index)
        sv = interaction_values_to_array(explainer.explain(x_val))

        assert len(sv) == len(X_train)
        assert np.sum(sv) == 1
        assert np.allclose(sv, 1 / len(y_train))

    def test_all_diffent_label_neighbors(self):
        """Tests NormalKNNExplainer behavior when a model is trained on only same label neighbors and k is equal to amount of training points.

        Expected behavior: All shapley values should be 0.
        """
        X_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        y_train = [0, 0, 0, 0]
        k = len(X_train)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        x_val = np.array([2, 2, 2])
        class_index = 1
        explainer = NormalKNNExplainer(model, class_index=class_index)
        sv = interaction_values_to_array(explainer.explain(x_val))

        assert len(sv) == len(X_train)
        assert np.sum(sv) == 0
        assert np.allclose(sv, 0)

    def test_agrees_with_brute_force(self):
        """Test that the result of NormalKNNExplainer agrees with that of the brute force implementation.

        The test is performed using part of the scikit-learn iris dataset.
        """
        iris = load_iris()
        X = cast("npt.NDArray[np.floating]", iris.data)
        y = cast("npt.NDArray[np.floating]", iris.target)

        train_max_size = 10
        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=41)
        X_train = X_train[:train_max_size]
        y_train = y_train[:train_max_size]

        x_val = X_test[0]

        n_classes = len(set(y))
        k = 3
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        for class_index in range(n_classes):
            brute_explainer = _BruteForceNormalKNNExplainer(model, class_index=class_index)
            iv_brute = interaction_values_to_array(brute_explainer.explain(x_val))

            jia_explainer = NormalKNNExplainer(model, class_index=class_index)
            iv_jia = interaction_values_to_array(jia_explainer.explain_function(x_val))

            assert np.allclose(iv_brute, iv_jia)

    def test_output_length(self):
        """Tests wether the explain_function() returns an output in a valid format."""
        dataf = load_iris()
        X = dataf.data
        y = dataf.target

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        probierModel = KNeighborsClassifier(n_neighbors=20)
        probierModel.fit(X_train, y_train)

        knn_expl = NormalKNNExplainer(model=probierModel, class_index=y_test[5])

        testoutput = knn_expl.explain_function(x=X_test[5])

        assert (len(testoutput)) == (len(y_train))

    def test_raises_on_invalid_weights(self):
        """Tests that instantiating the NormalKNNExplainer directly with an invalid value for weights raises an exception."""
        model = KNeighborsClassifier()
        model.fit(np.array([[0]]), np.array([0]))

        invalid_weights_values = ["invalid_weights", "distance"]

        for invalid_weights in invalid_weights_values:
            model.weights = invalid_weights

            with pytest.raises(ValueError, match=f"weights.*{invalid_weights}"):
                NormalKNNExplainer(model, class_index=0)

    def test_mode(self):
        """Tests that the explainer mode is set correctly."""
        knn_model = KNeighborsClassifier()
        knn_model.fit(np.array([[0]]), np.array([0]))

        explainer = NormalKNNExplainer(knn_model, class_index=0)
        assert explainer.mode == "normal"
