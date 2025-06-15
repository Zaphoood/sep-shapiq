"""Test the functionality of KNNExplainerBase."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn import KNNExplainerBase


def test_extract_training_data_from_model():
    """Tests that KNNExplainerBase succesfully extracts training data and the paramater k from a fitted model."""
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([0, 1])
    k = 11

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    knn_explainer = KNNExplainerBase(knn_model)

    assert np.allclose(knn_explainer.X_train, X_train)
    assert np.allclose(knn_explainer.y_train, y_train)
    assert knn_explainer.k == k


def test_raises_on_unfitted_model():
    """Tests that KNNExplainerBase raises an exception if its constructor is called with an unfitted model."""
    knn_model = KNeighborsClassifier()

    with pytest.raises(NotFittedError):
        KNNExplainerBase(knn_model)
