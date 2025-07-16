"""Implementation of Coalition Finding Algorithm."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
from shapiq.interaction_values import InteractionValues


# hatte ursprünglich max_size in den Klamemrn aber ich glaub wir brauchen doch max_order?
def get_coalitions(max_order: int) -> np.ndarray:
    """Returns all 2^n_players coalitions as a binary matrix (shape=(2^n, n_players)).

    Each row is a 0/1 vector of length n_players.
    """
    all_coals = list(product([0, 1], repeat=max_order))
    return np.array(all_coals, dtype=int)


def get_explanation_one_feature(
    coalition: np.ndarray, interaction_values: InteractionValues
) -> float:
    """Sums all e_i for individual players i ∈ S (|T|=1)."""
    total = 0.0
    S_known_indices = np.nonzero(coalition)[0]

    for i in S_known_indices:
        idx = interaction_values.interaction_lookup.get((i,))
        if idx is not None:
            total += interaction_values.values[idx]

    return total


def get_explanation_two_features(
    coalition: np.ndarray, interaction_values: InteractionValues
) -> float:
    """Sum the pairwise interaction values e_{i,j}.

    Calculated for each pair of active features {i,j} ⊆ S (|T|=2) in the coalition.
    """
    total = 0.0
    S_known_indices = np.nonzero(coalition)[0]

    for i, j in combinations(S_known_indices, 2):
        idx = interaction_values.interaction_lookup.get((i, j))
        if idx is None:  # just for safety
            idx = interaction_values.interaction_lookup.get((j, i))
        if idx is not None:
            total += interaction_values.values[idx]
    return total


def get_explanation_more_features(
    coalition: np.ndarray, interaction_values: InteractionValues
) -> float:
    """Sum all higher-order interaction values e_T for T of size 3 ≤ |T| ≤ max_order."""
    total = 0.0
    S_known_indices = np.nonzero(coalition)[0]
    max_order = interaction_values.max_order

    for k in range(3, max_order + 1):
        for subset in combinations(S_known_indices, k):
            idx = interaction_values.interaction_lookup.get(subset)
            if idx is not None:
                total += interaction_values.values[idx]
    return total


def subset_finding(
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
    max_coal: np.ndarray | None = None
    min_val = np.inf
    min_coal: np.ndarray | None = None

    e0 = interaction_values.baseline_value
    for coalition in get_coalitions(interaction_values.max_order):
        S_known = coalition.sum()

        if S_known == 0:
            total = e0

        elif S_known == 1:
            e1 = get_explanation_one_feature(coalition, interaction_values)
            total = e0 + e1

        elif S_known == 2:  # noqa: PLR2004
            e1 = get_explanation_one_feature(coalition, interaction_values)
            e2 = get_explanation_two_features(coalition, interaction_values)
            total = e0 + e1 + e2

        else:
            e1 = get_explanation_one_feature(coalition, interaction_values)
            e2 = get_explanation_two_features(coalition, interaction_values)
            e3 = get_explanation_more_features(coalition, interaction_values)
            total = e0 + e1 + e2 + e3

        if total > max_val:
            max_val = total
            max_coal = coalition.copy()
        if total < min_val:
            min_val = total
            min_coal = coalition.copy()

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
