"""Test the functionality of KNNClassifierExplainer."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from shapiq_student.explainer.knn import KNNClassifierExplainer
from shapiq_student.explainer.knn.base import interaction_values_to_array
from shapiq_student.explainer.knn.knn import BruteForceKNNClassifierExplainer


def test_minimal_example():
    """Tests the KNN Explainer with a minimal, hard-coded example by comparing to the brute force implementation."""
    X_train = np.array([[-3, 0], [-1, 0], [1, 0], [2, 0]])
    y_train = np.array([0, 1, 1, 1])
    x_val = np.array([0, 0])

    k = 3
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    for class_index in [0, 1]:
        brute_explainer = BruteForceKNNClassifierExplainer(model, class_index=class_index)
        iv_brute = interaction_values_to_array(brute_explainer.explain(x_val))

        jia_explainer = KNNClassifierExplainer(model, class_index=class_index)
        iv_jia = interaction_values_to_array(jia_explainer.explain_function(x_val))

        print(f"brute: {iv_brute}")  # noqa: T201
        print(f"jia: {iv_jia}")  # noqa: T201

        assert np.allclose(iv_brute, iv_jia)


def test_output_length():
    """Tests wether the explain_function() returns an output in a valid format."""
    dataf = load_iris()
    X = dataf.data
    y = dataf.target

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    probierModel = KNeighborsClassifier(n_neighbors=20)
    probierModel.fit(X_train, y_train)

    knn_expl = KNNClassifierExplainer(model=probierModel, class_index=y_test[5])

    testoutput = knn_expl.explain_function(x=X_test[5])

    assert (len(testoutput)) == (len(y_train))
