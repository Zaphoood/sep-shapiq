"""Implements the 'Solos' strategy for Coalition Finding."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

from shapiq_student.coalition_finder.util import min_max_coals_to_interaction_values


def coalition_finding(
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
        min_coal = []
        max_coal = []
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
