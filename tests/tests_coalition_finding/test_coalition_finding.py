"""Tests for the Coalition Finding algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from shapiq import InteractionValues

from shapiq_student.coalition_finder.benchmark import random_approximated_ivs_from_soums
from shapiq_student.coalition_finder.coalition_finder import (
    coalition_finding_exhaustive_search,
    compute_simplified_game_utility,
    subset_finding,
)
from shapiq_student.coalition_finder.strategies.equal_payoff import (
    coalition_finding as coalition_finding_equal_payoff,
    distribute_payoffs,
)
from shapiq_student.coalition_finder.util import get_min_max_from_interaction_values


@dataclass
class CoalitionFindingTestCase:
    """Defines a test case for the coalition finding algorithm."""

    max_size: int

    min_coal: tuple[int, ...]
    min_val: int
    max_coal: tuple[int, ...]
    max_val: int


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

        test_cases = [
            CoalitionFindingTestCase(
                max_size=0,
                min_coal=(),
                min_val=0,
                max_coal=(),
                max_val=0,
            ),
            CoalitionFindingTestCase(
                max_size=2,
                # (0, 1) = 6 + 3 + 2 + 0 = 11
                # (0, 2) = 5 + 3 + 2 + 0 = 10
                # (1, 2) = 3 + 2 + 2 + 0 = 7
                min_coal=(1, 2),
                min_val=7,
                max_coal=(0, 1),
                max_val=11,
            ),
        ]

        for test_case in test_cases:
            iv_found = coalition_finding_exhaustive_search(iv, max_size=test_case.max_size)
            (min_coal, min_val), (max_coal, max_val) = get_min_max_from_interaction_values(iv_found)

            assert min_coal == test_case.min_coal
            assert min_val == test_case.min_val
            assert max_coal == test_case.max_coal
            assert max_val == test_case.max_val

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


class TestBestIndividuals:
    """Tests the 'best individuals' approach for coalitiong finding."""

    def test_distribute_payoffs(self):
        """Tests that distributing higher-order interactions among their participants works correctly."""
        payoffs: dict[tuple[int, ...], float] = {
            (): 0,
            (0,): 3,
            (1,): 2,
            (2,): -1,
            (0, 1): 6,
            (0, 2): 5,
            (1, 2): 3,
            (0, 1, 2): 4,
        }
        expected_distributed_payoffs = np.array(
            [
                3 + 6 / 2 + 5 / 2 + 4 / 3,
                2 + 6 / 2 + 3 / 2 + 4 / 3,
                -1 + 5 / 2 + 3 / 2 + 4 / 3,
            ]
        )

        iv = interaction_values_from_payoffs(payoffs, index="SII")
        distributed_payoffs = distribute_payoffs(iv)

        assert np.allclose(distributed_payoffs, expected_distributed_payoffs)

    def test_coalition_finding_small(self):
        """Tests the 'best individuals' approach for coalition finding on a small, hand-picked example."""
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

        max_size = 2
        expected_min_coal = (1, 2)
        expected_max_coal = (0, 1)

        iv = interaction_values_from_payoffs(payoffs, index="SII")
        iv_min_max = coalition_finding_equal_payoff(iv, max_size=max_size)
        (min_coal, _), (max_coal, _) = get_min_max_from_interaction_values(iv_min_max)
        assert min_coal == expected_min_coal
        assert max_coal == expected_max_coal


@pytest.mark.parametrize("strategy", ["greedy", "solos", "equal_payoff"])
def test_all_strategies_can_be_called(strategy):
    """Tests that all strategies can be called."""
    soum = next(
        random_approximated_ivs_from_soums(
            n_games=1,
            n_players=10,
            n_basis_games=50,
            explanation_order=3,
            random_state=42,
            approximation_budget=100,
        )
    )
    subset_finding(soum, max_size=4, strategy=strategy)


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
