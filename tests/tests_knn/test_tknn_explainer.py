"""Test to compare all TKNN implementations."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

from shapiq_student.explainer.knn.base import interaction_values_to_array
from shapiq_student.explainer.knn.tknn import BruteForceTKNNExplainer, TKNNExplainer


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
