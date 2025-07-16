"""Implementation of Coalition Finding Algorithm."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from shapiq.interaction_values import InteractionValues


def compute_simplified_game_utility(
    coalition: tuple[int, ...], interaction_values: InteractionValues
) -> float:
    r"""Calculates the utility of a coalition in the simplified game constructed from the ``interaction_values``.

    The utility of the given coalition :math:`S` is calculated as

            :math:`\sum_{T\subseteq S, |T|\leq k} e_T`,

    where :math:`k` is the maximum order of the interaction values and :math:`e_T` is the interaction for subset :math:`T`.

    Args:
        coalition: The coalition in question.
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.

    Returns:
        The utility of the given coalition.
    """
    total = 0.0

    for k in range(interaction_values.max_order + 1):
        for subset in combinations(coalition, k):
            idx = interaction_values.interaction_lookup.get(subset)
            if idx is not None:
                total += interaction_values.values[idx]
    return total


def subset_finding(
    interaction_values: InteractionValues,
    max_size: int,
) -> InteractionValues:
    """Tries to find the maximizing and minimizing coalitions with size ``max_size`` for the given simplified game.

    Returns:
        An InteractionValues object containing the maximizing and minimizing coalitions together with their utilities.
    """
    return exhaustive_search(interaction_values, max_size)


def exhaustive_search(
    interaction_values: InteractionValues,
    max_size: int,
) -> InteractionValues:
    r"""Returns the maximizing and minimizing coalition of the given size of a simplified game :math:`\hat v_e` constructed from the given ``interaction_values`` along with their values.

    This is achieved by simply performing an exhaustive search of all coalitions with size ``max_size``.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        max_size: The size of the resulting maximizing and minimizing coalitions.

    Returns:
        An ``InteractionValues`` object containing the maximizing and minimizing coalitions.
    """
    if max_size < 0:
        msg = f"Parameter 'max_size' must be non-negative, but was {max_size}"
        raise ValueError(msg)

    max_val = -np.inf
    max_coal: tuple[int, ...] | None = None
    min_val = np.inf
    min_coal: tuple[int, ...] | None = None

    for coalition_indices in combinations(range(interaction_values.n_players), max_size):
        utility = compute_simplified_game_utility(coalition_indices, interaction_values)

        if utility > max_val:
            max_val = utility
            max_coal = coalition_indices
        if utility < min_val:
            min_val = utility
            min_coal = coalition_indices

    if max_coal is None or min_coal is None:
        msg = "Unreachable"
        raise RuntimeError(msg)

    max_coal_t = tuple(max_coal)
    min_coal_t = tuple(min_coal)
    interaction_lookup = {
        max_coal_t: 0,
        min_coal_t: 1,
    }
    values = np.zeros(2, dtype=float)
    values[interaction_lookup[max_coal_t]] = max_val
    values[interaction_lookup[min_coal_t]] = min_val

    return InteractionValues(
        values=values,
        index=interaction_values.index,
        n_players=interaction_values.n_players,
        max_order=max_size,
        min_order=max_size,
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )
