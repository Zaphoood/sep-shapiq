"""Utility function for Coalition Finding algorithms."""

from __future__ import annotations

from typing import cast

import numpy as np
from shapiq.interaction_values import InteractionValues


def min_max_coals_to_interaction_values(
    min_coal: tuple[int, ...],
    min_val: np.floating,
    max_coal: tuple[int, ...],
    max_val: np.floating,
    index: str,
    n_players: int,
) -> InteractionValues:
    """Wrap the given minimizing and maximizing coalitions in an ``InteractionValues`` object."""
    if len(min_coal) != len(max_coal):
        msg = f"Maximum and minimum coalition must be of the same size, but min. coal has size {len(min_coal)} and max. coal has size {len(max_coal)}."
        raise ValueError(msg)

    interaction_lookup = {
        max_coal: 0,
        min_coal: 1,
    }
    values = np.zeros(2, dtype=float)
    values[interaction_lookup[max_coal]] = max_val
    values[interaction_lookup[min_coal]] = min_val

    return InteractionValues(
        values=values,
        index=index,
        n_players=n_players,
        max_order=len(max_coal),
        min_order=len(max_coal),
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )


def get_min_max_from_interaction_values(
    iv: InteractionValues,
) -> tuple[tuple[tuple[int, ...], float], tuple[tuple[int, ...], float]]:
    """Given an InteractionValues object containing two coalitions, return a tuple with both coalitions as ``((min_coal, min_val), (max_coal, max_val))``."""
    expected_number_of_coalitions = [1, 2]
    n_coals = len(iv.interaction_lookup)
    if n_coals not in expected_number_of_coalitions:
        msg = f"Number of coalitions in InteractionValues must be one of: {', '.join(map(str, expected_number_of_coalitions))} coalitions but got {n_coals}"
        raise ValueError(msg)

    if n_coals == 1:
        (coal, coal_idx) = next(iter(iv.interaction_lookup.items()))
        coal_value = cast("float", iv.values[coal_idx])
        return (coal, coal_value), (coal, coal_value)

    (coal1, coal1_idx), (coal2, coal2_idx) = list(iv.interaction_lookup.items())
    coal1_value = iv.values[coal1_idx]
    coal2_value = iv.values[coal2_idx]

    if coal2_value > coal1_value:
        return (coal1, coal1_value), (coal2, coal2_value)
    return (coal2, coal2_value), (coal1, coal1_value)
