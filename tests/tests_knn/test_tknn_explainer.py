"""Test to compare all TKNN implementations."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier

from shapiq_student.explainer.knn.base import interaction_values_to_array
from shapiq_student.explainer.knn.tknn import BruteForceTKNNExplainer, TKNNExplainer


def test_extract_parameters_from_radius_model():
    """Tests that the TKNNExplainer succesfully extracts parameters from RadiusNeighborsClassifier model."""
    X_train = np.array([[1, 2, 3], [1, 2, 3]])
    y_train = np.array([0, 1])
    tau = 1

    radius_model = RadiusNeighborsClassifier(radius=tau)
    radius_model.fit(X_train, y_train)

    tknn_explainer = TKNNExplainer(radius_model, class_index=1)

    assert np.allclose(tknn_explainer.X_train, X_train)
    assert np.allclose(tknn_explainer.y_train, y_train)
    assert tknn_explainer.tau == tau


def test_no_neighbors_in_threshold():
    """Tests TKNNExplainer behavior when no neighbors are within threshold tau. Should return zero."""
    X_train = np.array([[10, 10, 10], [10, 10, 10]])
    y_train = np.array([0, 1])
    tau = 1
    x_val = np.array([1, 1, 1])
    radius_model = RadiusNeighborsClassifier(radius=tau)
    radius_model.fit(X_train, y_train)
    class_index = 0
    tknn_explainer = TKNNExplainer(radius_model, class_index=class_index)
    sv_array_tknn = interaction_values_to_array(tknn_explainer.explain(x_val))

    assert np.sum(sv_array_tknn) == 0


def test_with_load_iris():
    """Tests the correctness of the TKNN explainer by comparing its results to the baseline brute forece implementation.

    The model (RadiusNeighborsClassifier) is trained on the real-world, classic "Iris dataset".
    """
    iris = load_iris()
    X = cast("npt.NDArray[np.floating]", iris.data)
    y = cast("npt.NDArray[np.floating]", iris.target)

    train_max_size = 10
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.9, random_state=42)
    X_train = X_train[:train_max_size]
    y_train = y_train[:train_max_size]

    x_val = X_test[0]
    n_classes = len(set(y))

    tau = 1
    model = RadiusNeighborsClassifier(radius=tau)
    model.fit(X_train, y_train)

    for class_index in range(n_classes):
        brute_explainer = BruteForceTKNNExplainer(model, class_index=class_index)
        brute_iv = interaction_values_to_array(brute_explainer.explain(x_val))
        tknn_explainer = TKNNExplainer(model, class_index=class_index)
        tknn_iv = interaction_values_to_array(tknn_explainer.explain(x_val))

        assert np.allclose(brute_iv, tknn_iv)


def test_compare_tknn_brute_force():
    """Tests the correctness of the TKNN explainer by comparing its results to the baseline brute force implementation."""
    n_test_cases = 5

    seed = 42

    for _ in range(n_test_cases):
        rng = np.random.default_rng(seed=seed)
        seed += 1

        n_samples = 10
        X_train = rng.normal(size=(n_samples, 2))
        y_train = rng.integers(0, 2, size=n_samples)
        x_val = np.array([0, 0])
        tau = 1

        model = RadiusNeighborsClassifier(radius=tau)
        model.fit(X_train, y_train)

        for class_index in [0, 1]:
            explainer = TKNNExplainer(model, class_index)
            iv = interaction_values_to_array(explainer.explain(x_val))

            explainer_brute = BruteForceTKNNExplainer(model, class_index)
            iv_brute = interaction_values_to_array(explainer_brute.explain(x_val))

            assert np.allclose(iv, iv_brute)
