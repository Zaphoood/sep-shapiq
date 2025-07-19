"""Coalition finder module."""

from .benchmark import score_single_game
from .coalition_finder import (
    SubsetFindingStrategy,
    evaluate_all_coalitions,
    get_min_max_from_interaction_values,
    subset_finding,
)

__all__ = [
    "subset_finding",
    "SubsetFindingStrategy",
    "evaluate_all_coalitions",
    "get_min_max_from_interaction_values",
    "subset_finding",
    "score_single_game",
]
