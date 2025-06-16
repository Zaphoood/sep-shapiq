"""Example test case for the Explainer class."""

from __future__ import annotations

import pytest
from shapiq import ExactComputer
from shapiq.games.benchmark import SOUM

from shapiq_student import subset_finding


@pytest.mark.parametrize("max_size", [0, 1, 4, "n_players"])
def test_if_subset_finding_can_be_called(max_size):
    """Test if GaussianCopulaImputer is a subclass of shapiq's Imputer class."""
    game = SOUM(n=10, n_basis_games=50)
    computer = ExactComputer(n_players=game.n_players, game=game)
    interaction_values = computer(index="FSII", order=2)

    max_size = max_size if max_size != "n_players" else game.n_players
    output = subset_finding(interaction_values=interaction_values, max_size=max_size)
    assert output is not None
