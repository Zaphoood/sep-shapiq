"""A script that runs benchmarks on the coalition finding algorithms."""

from __future__ import annotations

import csv
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal

    from shapiq_student.coalition_finder.coalition_finder import SubsetFindingStrategy

from pathlib import Path

from shapiq import ExactComputer, InconsistentKernelSHAPIQ, InteractionValues
from shapiq.games.benchmark import SOUM

from shapiq_student.coalition_finder.benchmark import benchmark, score_single_game, time_strategy


def random_ivs_from_soums(
    n_games: int,
    n_players: int,
    n_basis_games: int,
    explanation_order: int,
    random_state: int,
) -> Iterator[InteractionValues]:
    """Creates an iterator over simplifed games generated from explanations of random Sum of Unanimity Games (SOUMs)."""
    for _ in range(n_games):
        game = SOUM(n=n_players, n_basis_games=n_basis_games, random_state=random_state)
        computer = ExactComputer(n_players=game.n_players, game=game)
        yield computer(index="FSII", order=explanation_order)
        random_state += 1


def random_approximated_ivs_from_soums(
    n_games: int,
    n_players: int,
    n_basis_games: int,
    explanation_order: int,
    random_state: int,
    approximation_budget: int,
) -> Iterator[InteractionValues]:
    """Creates an iterator over simplifed games generated from explanations of random Sum of Unanimity Games (SOUMs)."""
    for _ in range(n_games):
        game = SOUM(n=n_players, n_basis_games=n_basis_games, random_state=random_state)
        approximator = InconsistentKernelSHAPIQ(
            n=n_players, random_state=random_state, index="SII", max_order=explanation_order
        )
        yield approximator(budget=approximation_budget, game=game)
        random_state += 1


def benchmark_soums(
    strategies: list[SubsetFindingStrategy],
    n_games: int = 30,
    *,
    save_to_csv: bool = False,
    out_dir: str = ".",
) -> None:
    """Benchmarks the coalition finding algorithm using Sum of Unanimity Games."""
    n_players = 10
    n_basis_games = 50
    random_state = 42
    print(f"{n_games=}")
    print(f"{random_state=}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out_path = out_dir_path / f"{timestamp}_benchmark.csv"
    fieldnames = [
        "strategy",
        "n_games",
        "n_players",
        "n_basis_games",
        "coal_size",
        "random_state",
        "explanation_order",
        "avg_min_score",
        "avg_max_score",
        "t_delta",
    ]
    if save_to_csv:
        with out_path.open("a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for strategy in strategies:
        print(f"\n==== {strategy=} ====")
        for explanation_order in range(1, 6):
            for coal_size in range(3, 6):
                print(f"{explanation_order=}")
                print(f"{coal_size=}")
                avg_min_score, avg_max_score, avg_t_delta = benchmark(
                    strategy=strategy,
                    coal_size=coal_size,
                    ivs=random_ivs_from_soums(
                        n_games=n_games,
                        n_players=n_players,
                        n_basis_games=n_basis_games,
                        explanation_order=explanation_order,
                        random_state=random_state,
                    ),
                )
                print(f"avg score (min coal): {avg_min_score:.3f}")
                print(f"avg score (max coal): {avg_max_score:.3f}")
                print(f"avg t_delta: {avg_t_delta * 1000:.2f} ms")

                if save_to_csv:
                    with out_path.open("a") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(
                            {
                                "strategy": strategy,
                                "n_games": n_games,
                                "n_players": n_players,
                                "n_basis_games": n_basis_games,
                                "coal_size": coal_size,
                                "random_state": random_state,
                                "explanation_order": explanation_order,
                                "avg_min_score": avg_min_score,
                                "avg_max_score": avg_max_score,
                                "t_delta": avg_t_delta,
                            }
                        )


def time_soums_game_sizes(
    strategies: list[SubsetFindingStrategy],
    n_games: int = 30,
    max_players=15,
    players_step=1,
    *,
    save_to_csv: bool = False,
    out_dir: str = ".",
) -> None:
    """Benchmarks the coalition finding algorithm using Sum of Unanimity Games for a number of different game sizes."""
    n_basis_games = 50
    random_state = 42
    coal_size = 4
    explanation_order = 2
    print(f"{n_games=}")
    print(f"{random_state=}")
    print(f"{explanation_order=}")
    print(f"{coal_size=}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out_path = out_dir_path / f"{timestamp}_benchmark.csv"
    fieldnames = [
        "strategy",
        "n_games",
        "n_players",
        "n_basis_games",
        "coal_size",
        "random_state",
        "explanation_order",
        "avg_min_score",
        "avg_max_score",
        "t_delta",
    ]
    if save_to_csv:
        with out_path.open("a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for strategy in strategies:
        print(f"\n==== {strategy=} ====")
        for n_players in range(5, max_players + 1, players_step):
            print(f"{n_players=}")
            avg_t_delta = time_strategy(
                strategy=strategy,
                coal_size=coal_size,
                ivs=random_approximated_ivs_from_soums(
                    n_games=n_games,
                    n_players=n_players,
                    n_basis_games=n_basis_games,
                    explanation_order=explanation_order,
                    random_state=random_state,
                    approximation_budget=100,
                ),
            )
            print(f"avg t_delta: {avg_t_delta * 1000:.3f} ms")

            if save_to_csv:
                with out_path.open("a") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(
                        {
                            "strategy": strategy,
                            "n_games": n_games,
                            "n_players": n_players,
                            "n_basis_games": n_basis_games,
                            "coal_size": coal_size,
                            "random_state": random_state,
                            "explanation_order": explanation_order,
                            "avg_min_score": -1,
                            "avg_max_score": -1,
                            "t_delta": avg_t_delta,
                        }
                    )


def load_iv(*, instance: Literal["a", "b", "c"], large: bool) -> InteractionValues:
    """Load interaction values for a file.

    Args:
        instance: The instance to load.
        large: Whether to load the large or medium version of the interaction values.
            The large version has about 200 players, and the medium version has approximately
            50 players.

    Returns:
        The loaded interaction values.
    """
    path = Path(__file__).parent.parent / "tests_grading" / "data"
    size = "large" if large else "medium"
    path = path / f"iv_{instance}_{size}.pkl"

    iv = InteractionValues.load(path=str(path))
    return iv


def benchmark_precompute() -> None:
    """Benchmarks the coalition finding algorithm using precomputed games."""
    coal_sizes = [2, 3, 4, 5]
    instances: list[Literal["a", "b", "c"]] = ["a", "b", "c"]

    for instance in instances:
        for coal_size in coal_sizes:
            print(f"{instance=}")
            print(f"{coal_size=}")

            iv = load_iv(instance=instance, large=False)
            print(f"n_players: {iv.n_players}")

            min_score, max_score, t_delta = benchmark(
                strategy="equal_payoff",
                coal_size=coal_size,
                ivs=[iv],
            )
            print(f"score (min coal): {min_score:.3f}")
            print(f"score (max coal): {max_score:.3f}")
            print(f"t_delta: {t_delta:.3f}")


def single_soum(strategy: SubsetFindingStrategy) -> None:
    """Runs the given strategy on a single randomly generated SOUM game."""
    logging.basicConfig(level=logging.DEBUG)
    iv = next(
        random_ivs_from_soums(
            n_games=1, n_players=10, n_basis_games=50, explanation_order=3, random_state=42
        )
    )
    score_single_game(iv, strategy=strategy, coal_size=4)


def main() -> None:
    """The main entry point of the script."""
    time_soums_game_sizes(
        strategies=["solos", "equal_payoff", "greedy"],
        n_games=10,
        max_players=50,
        players_step=5,
        save_to_csv=True,
        out_dir="benchmark_results",
    )


if __name__ == "__main__":
    main()
