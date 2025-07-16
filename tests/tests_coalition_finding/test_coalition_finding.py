"""Tests for the Coalition Finding algorithm."""

from __future__ import annotations

import numpy as np
from shapiq import InteractionValues

from shapiq_student.coalition_finder.coalition_finder import (
    compute_simplified_game_utility,
    exhaustive_search,
)


class TestExhaustiveSearch:
    """Tests to ensure the correctness of the exhaustive-search algorithm which is used as a ground truth baseline."""

    def test_small(self):
        """Tests the correctness on a small, manually defined example."""
        payoffs: dict[tuple[int, ...], float] = {
            (): 0,
            (0,): 3,
            (1,): 2,
            (2,): 2,
            (0, 1): 6,
            (0, 2): 5,
            (1, 2): 3,
            (0, 1, 2): 4,
        }
        iv = interaction_values_from_payoffs(payoffs, index="SII")

        iv_found = exhaustive_search(iv, max_size=2)
        # (0, 1) = 6 + 3 + 2 + 0 = 11
        # (0, 2) = 5 + 3 + 2 + 0 = 10
        # (1, 2) = 3 + 2 + 2 + 0 = 7
        (min_coal, min_val), (max_coal, max_val) = get_min_max_from_interaction_values(iv_found)
        print(f"{min_coal=}")
        print(f"{max_coal=}")
        expected_min_coal = (1, 2)
        expected_min_val = 7
        expected_max_coal = (0, 1)
        expected_max_val = 11

        assert min_coal == expected_min_coal
        assert min_val == expected_min_val
        assert max_coal == expected_max_coal
        assert max_val == expected_max_val

    def test_compute_simplified_game_utility(self):
        """Tests that computing the utility of a coalition in the simplified game works."""
        payoffs: dict[tuple[int, ...], float] = {
            (): 0,
            (0,): 3,
            (1,): 2,
            (2,): 2,
            (0, 1): 6,
            (0, 2): 5,
            (1, 2): 3,
        }
        simplified_game = interaction_values_from_payoffs(payoffs, index="SII")

        expected_utilites = {
            (): 0,
            (0,): 3,
            (1,): 2,
            (2,): 2,
            (0, 1): 6 + 3 + 2,
            (0, 2): 5 + 3 + 2,
            (1, 2): 3 + 2 + 2,
            (0, 1, 2): 6 + 5 + 3 + 3 + 2 + 2,
        }
        for coal, expected_utility in expected_utilites.items():
            utility = compute_simplified_game_utility(coal, simplified_game)
            assert utility == expected_utility


def interaction_values_from_payoffs(
    payoffs: dict[tuple[int, ...], float], index: str
) -> InteractionValues:
    """Warp a dictionary defining the payoffs of a collaborative game in an ``InteractionValues`` object."""
    values = np.zeros(len(payoffs), dtype=float)
    interaction_lookup: dict[tuple[int, ...], int] = {}

    players = set()
    min_order = float("inf")
    max_order = 0
    for i, (coalition, payoff) in enumerate(payoffs.items()):
        values[i] = payoff
        interaction_lookup[tuple(sorted(coalition))] = i

        max_order = max(max_order, len(coalition))
        min_order = min(min_order, len(coalition))

        players.update(set(coalition))

    return InteractionValues(
        values=values,
        index=index,
        n_players=len(players),
        max_order=max_order,
        min_order=int(min_order),
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )


def get_min_max_from_interaction_values(
    iv: InteractionValues,
) -> tuple[tuple[tuple[int, ...], int], tuple[tuple[int, ...], int]]:
    """Given an InteractionValues object containing two coalitions, return a tuple with both coalitions as ``((min_coal, min_val), (max_coal, max_val))``."""
    expected_number_of_coalitions = 2
    if len(iv.interaction_lookup) != expected_number_of_coalitions:
        msg = f"InteractionValues object must contain exactly {expected_number_of_coalitions} coalitions but got {len(iv.interaction_lookup)}"
        raise ValueError(msg)

    (coal1, coal1_idx), (coal2, coal2_idx) = list(iv.interaction_lookup.items())
    coal1_value = iv.values[coal1_idx]
    coal2_value = iv.values[coal2_idx]

    if coal2_value > coal1_value:
        return (coal1, coal1_value), (coal2, coal2_value)
    return (coal2, coal2_value), (coal1, coal1_value)
