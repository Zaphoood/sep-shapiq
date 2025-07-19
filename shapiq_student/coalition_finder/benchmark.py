"""Provides benchmarking tools for coalition finding algorithms."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import comb
from shapiq import ExactComputer, Game, InteractionValues

from .coalition_finder import (
    SubsetFindingStrategy,
    evaluate_all_coalitions,
    get_min_max_from_interaction_values,
    subset_finding,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy.typing as npt


def score_single_game(
    interaction_values: InteractionValues,
    strategy: SubsetFindingStrategy,
    coal_size: int,
) -> tuple[npt.NDArray[np.floating], int, int]:
    r"""Evaluates a subset finding strategy on a single game by calculating the ranks of the estimated minimal and maximal coalitions among all coalitions of the given size.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        strategy: The coalition finding strategy to use.
        coal_size: The size of the resulting maximizing and minimizing coalitions.

    Returns:
        A tuple ``(utilities, min_rank, max_rank)``, where ``utilities`` is an array containing the utilites of all
        coalitions of size ``coal_size``, and ``min_rank`` and ``max_rank`` are the ranks of the estimated minimal and maximal coalitions respectively.
    """
    estimate = subset_finding(interaction_values, max_size=coal_size, strategy=strategy)
    (min_coal, min_val_est), (max_coal, max_val_est) = get_min_max_from_interaction_values(estimate)

    n_coalitions = comb(interaction_values.n_players, coal_size, exact=True)
    all_utilities = np.zeros(n_coalitions)

    min_val_actual: float | None = None
    max_val_actual: float | None = None

    for i, (coalition, utility) in enumerate(
        evaluate_all_coalitions(interaction_values, max_size=coal_size)
    ):
        all_utilities[i] = utility
        if coalition == min_coal:
            min_val_actual = utility
        if coalition == max_coal:
            max_val_actual = utility

    if min_val_actual is None or max_val_actual is None:
        msg = f"Unreachable: {min_val_actual=}, {max_val_actual=}"
        raise RuntimeError(msg)

    logging.debug(
        "min coal %s, estimated %f, actual %f", str(min_coal), min_val_est, min_val_actual
    )
    logging.debug(
        "max coal %s, estimated %f, actual %f", str(max_coal), max_val_est, max_val_actual
    )

    min_rank = int(np.sum(all_utilities <= min_val_actual))
    max_rank = int(np.sum(all_utilities <= max_val_actual))

    return all_utilities, min_rank, max_rank


def benchmark(
    strategy: SubsetFindingStrategy,
    games: Iterable[Game],
    explanation_order: int,
    coal_size: int,
) -> tuple[np.floating, np.floating]:
    """Evalutes the performance of a coalition finding strategy.

    This is achieved by executing the coalition finding algorithm on a given set of games and ranking the estimated
    minimal and maximal coalitions among all coalitions of the same size, which are evaluated via brute force search.

    For each game, the estimated maximal coalition is assigned a score according to the rank of its utility among all
    coalitions of size ``coal_size`` in the game: It gets a score of ``1`` if it is equal to the actual maximal coalition
    in the game and ``0`` if it is the minimal coalition. The estimated minimal coalition is scored analagously but inversely.
    Finally, the scores are averaged over all games.

    Args:
        strategy: The coalition finding strategy to evaluate.
        games: An iterable of ``Game``s to use for evaluation.
        explanation_order: The maximum order of the explanation from which the simplified game is constructed.
        coal_size: The size of the desired estimated minimal and maximal coalitions.

    Returns:
        A tuple of floats ``(avg_min_score, avg_max_score)``, where ``avg_min_score`` and ``avg_max_score`` are the
        average scores of the estimated minimal and maximal coalitions respectively.
    """
    min_scores = []
    max_scores = []
    for game in games:
        computer = ExactComputer(n_players=game.n_players, game=game)
        interaction_values = computer(index="FSII", order=explanation_order)

        utilities, min_rank, max_rank = score_single_game(
            interaction_values,
            strategy=strategy,
            coal_size=coal_size,
        )
        n_total = utilities.shape[0]

        min_score = (n_total - min_rank) / (n_total - 1)
        max_score = (max_rank - 1) / (n_total - 1)

        min_scores.append(min_score)
        max_scores.append(max_score)

    avg_min_score = np.mean(min_scores)
    avg_max_score = np.mean(max_scores)

    return avg_min_score, avg_max_score
