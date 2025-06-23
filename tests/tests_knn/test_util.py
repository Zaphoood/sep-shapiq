"""Tests the functionality of KNN utility functions."""

from __future__ import annotations

import numpy as np

from shapiq_student.explainer.knn.util import interaction_lookup_from_knn_shapley_values


def test_interaction_lookup_from_knn_shapley_values():
    """Tests that the values passed to interaction_lookup_from_knn_shapley_values are correct and parameters are adequately set."""
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

    iv = interaction_lookup_from_knn_shapley_values(sv)

    sv_reconstructed = [iv.values[iv.interaction_lookup[(i,)]] for i in range(n)]
    assert np.allclose(sv, sv_reconstructed)

    assert iv.min_order == 1
    assert iv.max_order == 1
    assert all(len(coalition) == 1 for coalition in iv.interaction_lookup)
    assert iv.n_players == n
