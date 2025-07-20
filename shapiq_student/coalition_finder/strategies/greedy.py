"""Implements the 'Greedy' strategy for Coalition Finding."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, TypeAlias

from shapiq_student.coalition_finder.util import min_max_coals_to_interaction_values

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

Coalition: TypeAlias = tuple[int, ...]


def coalition_finding(  # noqa: C901
    interaction_values: InteractionValues,
    max_size: int,
) -> InteractionValues:
    r"""Greedy search algorithm for finding the maximizing and minimizing coalitions with size ``max_size`` for the given simplified game.

    This algorithm starts with an empty set of players and keeps adding players that increase the total value by the
    biggest amount, until the desired coalition size is reached.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        max_size: The size of the resulting maximizing and minimizing coalitions.

    Returns:
        An InteractionValues object containing the maximizing and minimizing coalitions together with their utilities.
    """
    if max_size == 0:
        empty_coal = ()
        empty_val = interaction_values.values[interaction_values.interaction_lookup[empty_coal]]

        return min_max_coals_to_interaction_values(
            empty_coal,
            empty_val,
            empty_coal,
            empty_val,
            index=interaction_values.index,
            n_players=interaction_values.n_players,
        )

    interactions: list[tuple[Coalition, float]] = [
        (coal, interaction_values.values[value_idx])
        for coal, value_idx in interaction_values.interaction_lookup.items()
    ]

    # Maps each player to the indices of the coalitions it's part of
    affiliations = defaultdict(set)
    # Maps each possible coalition size to the indices of coalitions of that size
    coals_by_order = defaultdict(set)

    for i, (coalition, _) in enumerate(interactions):
        for player in coalition:
            affiliations[player].add(i)
        coals_by_order[len(coalition)].add(i)

    min_coal: list[int] = []
    empty_val = 0.0
    max_coal: list[int] = []
    max_val = 0.0
    while len(min_coal) < max_size:
        min_joiner: int | None = None
        min_increase = float("inf")
        max_joiner: int | None = None
        max_increase = float("-inf")
        for player in range(interaction_values.n_players):
            if player not in min_coal:
                increase = _get_joining_increase(
                    min_coal, player, affiliations, coals_by_order, interactions
                )
                if increase < min_increase:
                    min_increase = increase
                    min_joiner = player
            if player not in max_coal:
                increase = _get_joining_increase(
                    max_coal, player, affiliations, coals_by_order, interactions
                )
                if increase > max_increase:
                    max_increase = increase
                    max_joiner = player

        if min_joiner is None or max_joiner is None:
            msg = f"No players left to join at {min_coal=}, {max_coal=}. Total number of players is {interaction_values.n_players}."
            raise ValueError(msg)

        min_coal.append(min_joiner)
        empty_val += min_increase
        max_coal.append(max_joiner)
        max_val += max_increase

    return min_max_coals_to_interaction_values(
        tuple(min_coal),
        empty_val,
        tuple(max_coal),
        max_val,
        index=interaction_values.index,
        n_players=interaction_values.n_players,
    )


def _get_joining_increase(
    coalition: list[int],
    player: int,
    affiliations: dict[int, set[int]],
    coals_by_order: dict[int, set[int]],
    interactions: list[tuple[Coalition, float]],
) -> float:
    """Compute the increase in payoff when adding a given ``player`` to the ``coalition``."""
    new_coalition = {*coalition, player}
    increase = 0.0

    player_affiliations = affiliations[player]
    for order in range(1, len(new_coalition) + 1):
        coals_current_order = coals_by_order[order]
        for other_coalition_idx in player_affiliations.intersection(coals_current_order):
            other_coalition, _ = interactions[other_coalition_idx]

            if all(other_player in new_coalition for other_player in other_coalition):
                _, payoff = interactions[other_coalition_idx]
                increase += payoff

    return increase
