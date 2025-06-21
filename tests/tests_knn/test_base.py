"""Test the functionality of KNNExplainerBase."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn import (
    KNNExplainerBase,
    interaction_values_from_array,
)
from shapiq_student.explainer.knn.exceptions import MultiOutputKNNError


def test_extract_training_data_from_model():
    """Tests that KNNExplainerBase succesfully extracts training data and the paramater k from a fitted model."""
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([0, 1])
    k = 11

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    knn_explainer = KNNExplainerBase(knn_model, class_index=0)

    assert np.allclose(knn_explainer.X_train, X_train)
    assert np.allclose(knn_explainer.y_train, y_train)
    assert knn_explainer.k == k


def test_raises_on_unfitted_model():
    """Tests that KNNExplainerBase raises an exception if its constructor is called with an unfitted model."""
    knn_model = KNeighborsClassifier()

    with pytest.raises(NotFittedError):
        KNNExplainerBase(knn_model, class_index=0)


def test_raises_on_multi_output_model():
    """Tests that KNNExplainerBase raises an exception if its constructor is called with a model that has multiple output columns."""
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([[0, 1], [2, 3]])
    k = 11

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    with pytest.raises(MultiOutputKNNError):
        KNNExplainerBase(knn_model, class_index=0)


SHAPLEY_VALUES_INDEX = "SV"


def test_interaction_values_from_knn_shapley_values():
    """Tests that the values passed to interaction_values_from_knn_shapley_values are correct and parameters are adequately set."""
    sv = np.array(
        [
            -0.60893233,
            0.105666,
            0.7909356,
            1.00281216,
            1.24379742,
            -1.23121498,
            0.74966068,
            -0.82531311,
            0.87757074,
            -0.10162905,
        ]
    )
    n = sv.shape[0]

    iv = interaction_values_from_array(sv)

    sv_reconstructed = [iv.values[iv.interaction_lookup[(i,)]] for i in range(n)]
    assert np.allclose(sv, sv_reconstructed)

    assert iv.min_order == 1
    assert iv.max_order == 1
    assert all(len(coalition) == 1 for coalition in iv.interaction_lookup)
    assert iv.n_players == n
    assert iv.index == SHAPLEY_VALUES_INDEX
    assert iv.baseline_value == 0


def test_interaction_values_from_knn_shapley_values_emtpy():
    """Tests interaction_values_from_knn_shapley_values can handle empty arrays."""
    sv = np.array([])

    iv = interaction_values_from_array(sv)

    assert iv.values.shape[0] == 0

    assert iv.min_order == 1
    assert iv.max_order == 1
    assert iv.n_players == 0
    assert iv.index == SHAPLEY_VALUES_INDEX
    assert iv.interaction_lookup == {}
    assert iv.baseline_value == 0
