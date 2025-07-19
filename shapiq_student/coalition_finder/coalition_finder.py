"""Implementation of Coalition Finding Algorithm."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Literal, cast

from shapiq_student.coalition_finder.util import min_max_coals_to_interaction_values

if TYPE_CHECKING:
    import numpy.typing as npt

from itertools import combinations

import numpy as np
from shapiq.interaction_values import InteractionValues


def evaluate_all_coalitions(
    interaction_values: InteractionValues,
    max_size: int,
) -> Iterator[tuple[tuple[int, ...], float]]:
    r"""Evaluates the utility of all coalitions of a given size of a simplified game :math:`\hat v_e`.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        max_size: The size of the resulting maximizing and minimizing coalitions.

    Yields:
        A tuple ``(coalition, utility)``, where ``coalition`` is an integer tuple of length ``max_size``.
    """
    if max_size < 0:
        msg = f"Parameter 'max_size' must be non-negative, but was {max_size}"
        raise ValueError(msg)

    for coalition_indices in combinations(range(interaction_values.n_players), max_size):
        utility = compute_simplified_game_utility(coalition_indices, interaction_values)
        yield coalition_indices, utility


def coalition_finding_exhaustive_search(
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
    max_value = -np.inf
    max_coalition: tuple[int, ...] | None = None
    min_value = np.inf
    min_coalition: tuple[int, ...] | None = None

    for coalition, utility in evaluate_all_coalitions(interaction_values, max_size):
        if utility > max_value:
            max_value = utility
            max_coalition = coalition
        if utility < min_value:
            min_value = utility
            min_coalition = coalition

    if max_coalition is None or min_coalition is None:
        msg = "Unreachable"
        raise RuntimeError(msg)

    return min_max_coals_to_interaction_values(
        min_coal=min_coalition,
        min_val=min_value,
        max_coal=max_coalition,
        max_val=max_value,
        index=interaction_values.index,
        n_players=interaction_values.n_players,
    )


def coalition_finding_equal_payoff(
    interaction_values: InteractionValues,
    max_size: int,
) -> InteractionValues:
    r"""Heuristic algorithm for finding the maximizing and minimizing coalitions with size ``max_size`` for the given simplified game.

    This algorithm works by first distributing all interactions of order :math:`k \geq 2` equally to their participants and then
    chooses the best (worst) :math:`\ell` players as the maximizing (minimizing) coalition.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        max_size: The size of the resulting maximizing and minimizing coalitions.

    Returns:
        An InteractionValues object containing the maximizing and minimizing coalitions together with their utilities.
    """
    if max_size == 0:
        min_coal = ()
        max_coal = ()
        min_val = cast(
            "np.floating",
            interaction_values.values[interaction_values.interaction_lookup[()]],
        )
        max_val = min_val
    else:
        player_scores = distribute_payoffs(interaction_values)
        # Sorts player indices by their scores
        players_sorted = np.argsort(player_scores)

        min_coal = sorted(players_sorted[:max_size])
        max_coal = sorted(players_sorted[-max_size:])
        # Return estimates of coalition utilities, not their real utility
        min_val = np.sum(player_scores[min_coal])
        max_val = np.sum(player_scores[max_coal])

    return min_max_coals_to_interaction_values(
        tuple(min_coal),
        min_val,
        tuple(max_coal),
        max_val,
        index=interaction_values.index,
        n_players=interaction_values.n_players,
    )


def distribute_payoffs(interaction_values: InteractionValues) -> npt.NDArray[np.floating]:
    """Distribute payoffs of all coalitions equally among their participants.

    Args:
        interaction_values: The payoffs to distribute.

    Returns:
        An ``np.ndarray`` of shape ``(n_players,)`` containing the distributed payoff for each player.
    """
    player_scores = np.zeros(interaction_values.n_players, dtype=np.float64)

    for coalition, value_idx in interaction_values.interaction_lookup.items():
        coalition_size = len(coalition)
        if coalition_size == 0:
            continue

        coalition_value = interaction_values.values[value_idx]
        per_player_payoff = coalition_value / coalition_size

        for player in coalition:
            player_scores[player] += per_player_payoff

    return player_scores


def coalition_finding_solos(
    interaction_values: InteractionValues,
    max_size: int,
) -> InteractionValues:
    r"""Heuristic algorithm for finding the maximizing and minimizing coalitions with size ``max_size`` for the given simplified game.

    This algorithm works by choosing the players with the lowest (highest) Shapley value as the maximal (minimal) coalition.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        max_size: The size of the resulting maximizing and minimizing coalitions.

    Returns:
        An InteractionValues object containing the maximizing and minimizing coalitions together with their utilities.
    """
    if max_size == 0:
        min_coal = ()
        max_coal = ()
        min_val = cast(
            "np.floating",
            interaction_values.values[interaction_values.interaction_lookup[()]],
        )
        max_val = min_val
    else:
        player_svs = np.zeros(interaction_values.n_players, dtype=float)
        for player in range(interaction_values.n_players):
            player_svs[player] = interaction_values.values[
                interaction_values.interaction_lookup[(player,)]
            ]

        players_sorted = player_svs.argsort()

        min_coal = sorted(players_sorted[:max_size])
        max_coal = sorted(players_sorted[-max_size:])
        # Return estimates of coalition utilities, not their real utility
        min_val = np.sum(player_svs[min_coal])
        max_val = np.sum(player_svs[max_coal])

    return min_max_coals_to_interaction_values(
        tuple(min_coal),
        min_val,
        tuple(max_coal),
        max_val,
        index=interaction_values.index,
        n_players=interaction_values.n_players,
    )


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


SubsetFindingStrategy = Literal["exhaustive_search", "equal_payoff", "solos"]
SubsetFindingFunction = Callable[[InteractionValues, int], InteractionValues]
SUBSET_FINDING_STRATEGIES: dict[SubsetFindingStrategy, SubsetFindingFunction] = {
    "exhaustive_search": coalition_finding_exhaustive_search,
    "equal_payoff": coalition_finding_equal_payoff,
    "solos": coalition_finding_solos,
}


def subset_finding(
    interaction_values: InteractionValues,
    max_size: int,
    strategy: SubsetFindingStrategy = "equal_payoff",
) -> InteractionValues:
    r"""Tries to find the maximizing and minimizing coalitions with size ``max_size`` for the given simplified game.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        max_size: The size of the resulting maximizing and minimizing coalitions.
        strategy: The strategy to use for finding the coalitions. Defaults to 'equal_payoff'.

    Returns:
        An InteractionValues object containing the maximizing and minimizing coalitions together with their utilities.
    """
    subset_finding_fn = SUBSET_FINDING_STRATEGIES.get(strategy)
    if subset_finding_fn is None:
        msg = (
            f"Invalid subset finding strategy: '{strategy}'. Available strategies are: "
            + ", ".join(SUBSET_FINDING_STRATEGIES)
        )
        raise ValueError(msg)

    return subset_finding_fn(interaction_values, max_size)
