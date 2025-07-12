"""Implementation of Coalition Finding Algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


def subset_finding(interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """Find the coalition with maximum interaction value for a given max_size."""
    max_value = float("-inf")
    best_coalition = None

    for coalition, value_idx in interaction_values.interaction_lookup.items():
        if len(coalition) == max_size and interaction_values.values[value_idx] > max_value:
            max_value = interaction_values.values[value_idx]
            best_coalition = coalition

    return best_coalition
