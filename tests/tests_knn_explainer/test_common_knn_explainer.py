"""Test the functionality of the private _CommonKNNExplainer class."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn._common_knn import _CommonKNNExplainer
from shapiq_student.explainer.knn.base import KNNExplainer
from shapiq_student.explainer.knn.exceptions import UnsupportedKNNWeightsError
from shapiq_student.explainer.knn.normal_knn import NormalKNNExplainer
from shapiq_student.explainer.knn.weighted_knn import WeightedKNNExplainer


def test_raises_on_unsupported_weights_value():
    """Tests that an exception is raised when instantiating the explainer with a model that has an unsupported weights parameter."""
    X_train = np.array([[0]])
    y_train = np.array([0])

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    invalid_weights = "invalid_weights"
    model.weights = invalid_weights

    with pytest.raises(UnsupportedKNNWeightsError):
        KNNExplainer(model, class_index=0)


def test_select_explainer():
    """Tests that _CommonKNNExplainer automagically turns into the right explainer class for a given weights parameter."""
    test_cases = [
        ("uniform", NormalKNNExplainer),
        ("distance", WeightedKNNExplainer),
    ]

    for weights, expected_explainer_class in test_cases:
        model = KNeighborsClassifier(weights=weights)
        model.fit(np.array([[0]]), np.array([0]))
        explainer = _CommonKNNExplainer(model, class_index=0)
        assert isinstance(explainer, expected_explainer_class)
