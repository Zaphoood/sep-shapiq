"""Test the functionality of KNNExplainerBase."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from shapiq import InteractionValues
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from shapiq_student.explainer.knn import (
    KNNExplainer,
    interaction_values_from_array,
)
from shapiq_student.explainer.knn._common_knn import _CommonKNNExplainer
from shapiq_student.explainer.knn.base import interaction_values_to_array
from shapiq_student.explainer.knn.exceptions import MultiOutputKNNError
from shapiq_student.explainer.knn.threshold_nn import ThresholdNNExplainer


def test_extract_training_data_from_model():
    """Tests that KNNExplainerBase succesfully extracts training data and the paramater k from a fitted model."""
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([0, 1])
    k = 11
    class_index = 0

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    knn_explainer = KNNExplainer(knn_model, class_index=class_index)

    assert np.allclose(knn_explainer.X_train, X_train)
    assert np.allclose(knn_explainer.y_train, y_train)
    assert set(knn_explainer.y_train_classes) == set(y_train)
    assert knn_explainer.k == k
    assert knn_explainer.class_index == class_index


def test_raises_on_unfitted_model():
    """Tests that KNNExplainerBase raises an exception if its constructor is called with an unfitted model."""
    knn_model = KNeighborsClassifier()

    with pytest.raises(NotFittedError):
        KNNExplainer(knn_model, class_index=0)


def test_raises_on_multi_output_model():
    """Tests that KNNExplainerBase raises an exception if its constructor is called with a model that has multiple output columns."""
    X_train = np.array([[1, 2, 3], [4, 5, 6]])
    y_train = np.array([[0, 1], [2, 3]])
    k = 11

    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    with pytest.raises(MultiOutputKNNError):
        KNNExplainer(knn_model, class_index=0)


def test_class_index_none():
    """Tests that setting ``class_index=0`` in the constructor is handled correctly."""
    knn_model = KNeighborsClassifier()
    knn_model.fit(np.array([[0]]), np.array([0]))

    explainer = KNNExplainer(knn_model, class_index=None)
    assert explainer.class_index == 1


def test_select_explainer_class():
    """Tests KNNExplainer automagically selects the right explainer for a given model."""
    test_cases = [
        (KNeighborsClassifier(), _CommonKNNExplainer),
        (RadiusNeighborsClassifier(), ThresholdNNExplainer),
    ]

    for model, expected_explainer_class in test_cases:
        model.fit(np.array([[0]]), np.array([0]))
        explainer = KNNExplainer(model)
        assert isinstance(explainer, expected_explainer_class)


SHAPLEY_VALUES_INDEX = "SV"


def test_interaction_values_from_array():
    """Tests that the values passed to interaction_values_from_array are correct and parameters are adequately set."""
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


def test_interaction_values_from_array_empty():
    """Tests that interaction_values_from_array can handle empty arrays."""
    sv = np.array([])

    iv = interaction_values_from_array(sv)

    assert iv.values.shape[0] == 0

    assert iv.min_order == 1
    assert iv.max_order == 1
    assert iv.n_players == 0
    assert iv.index == SHAPLEY_VALUES_INDEX
    assert iv.interaction_lookup == {}
    assert iv.baseline_value == 0


def test_interaction_values_to_array():
    """Tests that interaction_values_to_array succesfully transforms InteractionValues back to an ordered array."""
    n_players = 10
    rng = np.random.default_rng(seed=123)
    sv = rng.normal(size=(n_players,))
    permutation = rng.permutation(n_players)
    sv_permuted = np.zeros_like(sv)
    sv_permuted[permutation] = sv

    lookup = {(i,): cast("int", i_unperm) for i, i_unperm in enumerate(permutation)}

    iv = InteractionValues(
        sv_permuted,
        index="SV",
        max_order=1,
        min_order=1,
        n_players=n_players,
        baseline_value=0,
        interaction_lookup=lookup,
    )

    sv_reconstructed = interaction_values_to_array(iv)
    assert np.allclose(sv_reconstructed, sv)
