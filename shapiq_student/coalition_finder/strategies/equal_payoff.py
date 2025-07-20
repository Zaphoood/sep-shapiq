"""Implements the 'Equal Payoff' strategy for Coalition Finding."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from shapiq_student.coalition_finder.util import min_max_coals_to_interaction_values

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq.interaction_values import InteractionValues


import numpy as np


def coalition_finding(
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

        min_coal = sorted(players_sorted[:max_size])  # type: ignore[type-var,assignment]
        max_coal = sorted(players_sorted[-max_size:])  # type: ignore[type-var,assignment]
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
