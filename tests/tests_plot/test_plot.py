"""Tests for the plotting module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from shapiq_student.plot import plot_points_shapley_2d


def test_plot_points_shapley_2d_can_be_called():
    """Tests that plot_points_shapley_2d can be called."""
    _, ax = plt.subplots()

    n_features = 2
    X_train, y_train = make_classification(
        n_samples=30,
        n_features=n_features,
        n_redundant=0,
        n_clusters_per_class=1,
        n_informative=2,
        n_classes=2,
        random_state=45,
    )
    x_test = np.zeros(n_features)
    classes = np.array(list(set(y_train)))

    rng = np.random.default_rng(seed=100)
    sv = rng.uniform(-1, 1, size=(X_train.shape[0],))
    plot_points_shapley_2d(
        ax, X_train, y_train, sv, classes, x_test=x_test, show_max=True, title="Title"
    )
