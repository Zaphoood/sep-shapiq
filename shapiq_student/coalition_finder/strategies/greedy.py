"""Implements the 'Greedy' strategy for Coalition Finding."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


def coalition_finding(
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
    raise NotImplementedError
