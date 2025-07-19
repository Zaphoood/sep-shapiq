"""A script that runs benchmarks on the coalition finding algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
from shapiq.games.benchmark import SOUM

from shapiq_student.coalition_finder.benchmark import benchmark


def generate_random_soums(
    n_games: int,
    n_players: int,
    n_basis_games: int,
    random_state: int,
) -> Iterator[SOUM]:
    """Creates an iterator over randomly generated Sum of Unanimity Games (SOUMs)."""
    for _ in range(n_games):
        yield SOUM(n=n_players, n_basis_games=n_basis_games, random_state=random_state)
        random_state += 1


def main() -> None:
    """The main entry point of the script."""
    n_games = 10
    random_state = 42
    print(f"{n_games=}")
    print(f"{random_state=}")
    print()

    for explanation_order in range(1, 6):
        print(f"{explanation_order=}")
        avg_min_score, avg_max_score = benchmark(
            strategy="best_individuals",
            coal_size=2,
            games=generate_random_soums(
                n_games=n_games, n_players=10, n_basis_games=50, random_state=random_state
            ),
            explanation_order=explanation_order,
        )
        print(f"avg score (min coal): {avg_min_score:.3f}")
        print(f"avg score (max coal): {avg_max_score:.3f}")


if __name__ == "__main__":
    main()
