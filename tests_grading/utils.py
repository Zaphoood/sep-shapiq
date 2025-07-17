"""Utility functions for the grading tests."""

from __future__ import annotations

import numpy as np


def get_random_coalitions(
    n_features: int, n_coalitions: int = 5, random_seed: int | None = None
) -> np.ndarray:
    """Generate random coalitions for testing.

    Args:
        n_features: Number of features in the dataset.
        n_coalitions: Number of coalitions to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        A boolean array representing the coalitions.
    """
    rng = np.random.default_rng(random_seed)
    return rng.choice([True, False], size=(n_coalitions, n_features))
