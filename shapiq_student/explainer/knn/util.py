"""Common utility functions used in the implementations of KNN Explainers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def interaction_lookup_from_knn_shapley_values(
    shapley_values: npt.NDArray[np.floating],
) -> InteractionValues:
    """Convert an array of Shapley Values to a `shapiq.interaction_values.InteractionValues` object.

    Args:
        shapley_values: A np.ndarray containing the Shapley Value of the ith training point at index i

    Returns:
        An InteractionValues object containing the provided Shapley Values with an appropriate `interaction_lookup` dict and with `min_order==max_order==1` set.
    """
    n_players = shapley_values.shape[0]
    interaction_lookup: dict[tuple[int, ...], int] = {(i,): i for i in range(n_players)}

    return InteractionValues(
        shapley_values,
        "SII",
        min_order=1,
        max_order=1,
        n_players=n_players,
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )
