"""Test the functionality of the private _CommonKNNExplainer class."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn.base import KNNExplainer
from shapiq_student.explainer.knn.exceptions import UnsupportedKNNWeightsError


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
