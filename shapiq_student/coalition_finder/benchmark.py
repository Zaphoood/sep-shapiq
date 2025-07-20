"""Provides benchmarking tools for coalition finding algorithms."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import comb

from .coalition_finder import (
    SubsetFindingStrategy,
    compute_simplified_game_utility,
    evaluate_all_coalitions,
    subset_finding,
)
from .util import get_min_max_from_interaction_values

if TYPE_CHECKING:
    from collections.abc import Iterable

    from shapiq import InteractionValues


def score_single_game(
    interaction_values: InteractionValues,
    strategy: SubsetFindingStrategy,
    coal_size: int,
) -> tuple[int, int, int, float]:
    r"""Evaluates a subset finding strategy on a single game by calculating the ranks of the estimated minimal and maximal coalitions among all coalitions of the given size.

    Args:
        interaction_values: The interaction values from which the simplified game :math:`\hat v_e` is constructed.
        strategy: The coalition finding strategy to use.
        coal_size: The size of the resulting maximizing and minimizing coalitions.

    Returns:
        A tuple ``(n_coalitions, min_rank, max_rank, t_delta)``, where ``n_coalitions`` is the number of all coalitions of size ``coal_size``,
        ``min_rank`` and ``max_rank`` are the ranks of the estimated minimal and maximal coalitions respectively,
        and ``t_delta`` is the time it took to run the estimation algorithm.
    """
    t_start = time.time()
    estimate = subset_finding(interaction_values, max_size=coal_size, strategy=strategy)
    t_end = time.time()
    t_delta = t_end - t_start

    (min_coal, min_val_est), (max_coal, max_val_est) = get_min_max_from_interaction_values(estimate)

    n_coalitions = comb(interaction_values.n_players, coal_size, exact=True)

    min_val_actual: float = compute_simplified_game_utility(min_coal, interaction_values)
    max_val_actual: float = compute_simplified_game_utility(max_coal, interaction_values)

    min_rank = 0
    max_rank = 0
    for _, utility in evaluate_all_coalitions(interaction_values, max_size=coal_size):
        min_rank += utility <= min_val_actual
        max_rank += utility <= max_val_actual

    logging.debug(
        "min coal %s, estimated %f, actual %f", str(min_coal), min_val_est, min_val_actual
    )
    logging.debug(
        "max coal %s, estimated %f, actual %f", str(max_coal), max_val_est, max_val_actual
    )

    return n_coalitions, min_rank, max_rank, t_delta


def benchmark(
    strategy: SubsetFindingStrategy,
    ivs: Iterable[InteractionValues],
    coal_size: int,
) -> tuple[np.floating, np.floating, np.floating]:
    """Evalutes the performance of a coalition finding strategy.

    This is achieved by executing the coalition finding algorithm on a given set of simplified games and ranking the estimated
    minimal and maximal coalitions among all coalitions of the same size, which are evaluated via brute force search.

    For each game, the estimated maximal coalition is assigned a score according to the rank of its utility among all
    coalitions of size ``coal_size`` in the game: It gets a score of ``1`` if it is equal to the actual maximal coalition
    in the game and ``0`` if it is the minimal coalition. The estimated minimal coalition is scored analagously but inversely.
    Finally, the scores are averaged over all games.

    Args:
        strategy: The coalition finding strategy to evaluate.
        ivs: An iterable of ``InteractionValues``s to use as simplified games for evaluation.
        explanation_order: The maximum order of the explanation from which the simplified game is constructed.
        coal_size: The size of the desired estimated minimal and maximal coalitions.

    Returns:
        A tuple of floats ``(avg_min_score, avg_max_score)``, where ``avg_min_score`` and ``avg_max_score`` are the
        average scores of the estimated minimal and maximal coalitions respectively.
    """
    min_scores = []
    max_scores = []
    t_deltas = []
    for iv in ivs:
        n_total, min_rank, max_rank, t_delta = score_single_game(
            iv, strategy=strategy, coal_size=coal_size
        )
        t_deltas.append(t_delta)

        min_score = (n_total - min_rank) / (n_total - 1)
        max_score = (max_rank - 1) / (n_total - 1)

        min_scores.append(min_score)
        max_scores.append(max_score)

    avg_min_score = np.mean(min_scores)
    avg_max_score = np.mean(max_scores)
    avg_t_delta = np.mean(t_deltas)

    return avg_min_score, avg_max_score, avg_t_delta


def time_strategy(
    strategy: SubsetFindingStrategy,
    ivs: Iterable[InteractionValues],
    coal_size: int,
) -> np.floating:
    """Measures the execution time of a coalition finding strategy across multiple runs.

    Args:
        strategy: The coalition finding strategy to evaluate.
        ivs: An iterable of ``InteractionValues``s to use as simplified games for evaluation.
        coal_size: The size of the desired estimated minimal and maximal coalitions.

    Returns:
        The average execution time of the coalition finding algorithm across all simplified games
    """
    t_deltas = []
    for iv in ivs:
        t_start = time.time()
        subset_finding(iv, max_size=coal_size, strategy=strategy)
        t_end = time.time()
        t_delta = t_end - t_start
        t_deltas.append(t_delta)

    avg_t_delta = np.mean(t_deltas)

    return avg_t_delta
