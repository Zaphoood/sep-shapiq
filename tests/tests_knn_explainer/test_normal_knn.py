"""Test the functionality of KNNClassifierExplainer."""

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


def test_agrees_with_brute_force():
    """Test that the result of KNNExplainer agrees with that of the brute force implementation.

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

    knn_expl = NormalKNNExplainer(model=probierModel, class_index=y_test[5])

    testoutput = knn_expl.explain_function(x=X_test[5])

    assert (len(testoutput)) == (len(y_train))


def test_raises_on_invalid_weights():
    """Tests that instantiating the NormalKNNExplainer directly with an invalid value for weights raises an exception."""
    model = KNeighborsClassifier()
    model.fit(np.array([[0]]), np.array([0]))

    invalid_weights_values = ["invalid_weights", "distance"]

    for invalid_weights in invalid_weights_values:
        model.weights = invalid_weights

        with pytest.raises(ValueError, match=f"weights.*{invalid_weights}"):
            NormalKNNExplainer(model, class_index=0)
