"""Implementation of Coalition Finding Algorithm."""

from __future__ import annotations

from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_coalitions(n_players: int) -> Iterator[tuple[bool, ...]]:
    """Returns all 2^n_players coalitions as a binary matrix (shape=(2^n, n_players)).

    Each row is a 0/1 vector of length n_players.
    """
    return product([False, True], repeat=n_players)


def compute_simplified_game_utility(
    coalition: tuple[int, ...], simplified_game: InteractionValues
) -> float:
    """Calculates the utility of a given coalition based in the simplified game."""
    total = 0.0

    for k in range(simplified_game.max_order + 1):
        for subset in combinations(coalition, k):
            idx = simplified_game.interaction_lookup[subset]
            total += simplified_game.values[idx]
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
    r"""Searches all S ⊆ N with |S|=coalition_size.

    Returns the coalition that maximizes \hat v_e(S) and the one that minimizes it,
    along with their values.

    Returns:
        An InteractionValues object containing the maximizing and minimizing coalitions.
    """
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
        error_msg = "no Koalition found!"
        raise ValueError(error_msg)

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
        n_players=len(max_coal),
        max_order=max_size,
        min_order=max_size,
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )
