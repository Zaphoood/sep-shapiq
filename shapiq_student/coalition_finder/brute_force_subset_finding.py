"""Brute Force implementation of subset/coalition finding."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from shapiq import InteractionValues


def brute_force_subset_finding(
    interaction_values: InteractionValues,
    coalition_size: int,
) -> InteractionValues:
    """Finds the coalitions of size coalition_size with minimum and maximum interaction value.

    Args:
        interaction_values (InteractionValues): The input interaction value object.
        coalition_size (int): The coalition size to consider.

    Returns:
        InteractionValues: New Interaction Values object with only the min and max coalitions of size coalition_size.
    """
    n_players = interaction_values.n_players
    min_val = float("inf")
    max_val = float("-inf")
    min_coalition: tuple[int, ...] | None = None
    max_coalition: tuple[int, ...] | None = None

    for coalition in combinations(range(n_players), coalition_size):
        idx = interaction_values.interaction_lookup.get(coalition)
        if idx is None:
            continue
        val = interaction_values.values[idx]
        if val < min_val:
            min_val = val
            min_coalition = coalition
        if val > max_val:
            max_val = val
            max_coalition = coalition

    coalitions = []
    values = []
    interaction_lookup = {}

    if min_coalition is not None:
        coalitions.append(min_coalition)
        values.append(min_val)
        interaction_lookup[min_coalition] = 0
        # TODO(murscht): add some kind of statement for when min and max are same (maybe just append anyways and explain possible doubling in docstring)
    if max_coalition is not None and max_coalition != min_coalition:
        coalitions.append(max_coalition)
        values.append(max_val)
        interaction_lookup[max_coalition] = len(values) - 1

    return InteractionValues(
        values=np.array(values),
        interaction_lookup=interaction_lookup,
        n_players=n_players,
        min_order=coalition_size,
        max_order=coalition_size,
        baseline_value=interaction_values.baseline_value,
        index=interaction_values.index,
    )
