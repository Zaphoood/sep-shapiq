"""A script that runs benchmarks on the coalition finding algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
from shapiq import ExactComputer, InteractionValues
from shapiq.games.benchmark import SOUM

from shapiq_student.coalition_finder.benchmark import benchmark


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


def main() -> None:
    """The main entry point of the script."""
    n_games = 10
    n_players = 10
    coal_size = 4
    random_state = 42
    print(f"{n_games=}")
    print(f"{coal_size=}")
    print(f"{random_state=}")
    print()

    for explanation_order in range(1, 6):
        print(f"{explanation_order=}")
        avg_min_score, avg_max_score = benchmark(
            strategy="best_individuals",
            coal_size=coal_size,
            ivs=random_ivs_from_soums(
                n_games=n_games,
                n_players=n_players,
                n_basis_games=50,
                explanation_order=explanation_order,
                random_state=random_state,
            ),
        )
        print(f"avg score (min coal): {avg_min_score:.3f}")
        print(f"avg score (max coal): {avg_max_score:.3f}")


if __name__ == "__main__":
    main()
