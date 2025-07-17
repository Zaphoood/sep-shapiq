"""Example test case for the Explainer class."""

from __future__ import annotations

import logging
import pathlib
from typing import Literal

import pytest
from shapiq import ExactComputer, InteractionValues
from shapiq.games.benchmark import SOUM

from shapiq_student import subset_finding


def load_iv(*, instance: Literal["a", "b", "c"], large: bool) -> InteractionValues:
    """Load interaction values for testing subset finding.

    Args:
        instance: The instance to load.
        large: Whether to load the large or medium version of the interaction values.
            The large version has about 200 players, and the medium version has approximately
            50 players.

    Returns:
        The loaded interaction values.
    """
    path = pathlib.Path(__file__).parent / "data"
    size = "large" if large else "medium"
    path = path / f"iv_{instance}_{size}.pkl"
    return InteractionValues.load(path=path)


@pytest.mark.parametrize("instance", ["a", "b", "c"])
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("max_size", [2, 5, 10])
def test_subset_finding(*, instance: Literal["a", "b", "c"], large: bool, max_size: int):
    """Test if subset finding works with precomputed interaction values."""
    interaction_values = load_iv(instance=instance, large=large)
    n_players = interaction_values.n_players
    message = f"max_size={max_size}, n_players={n_players}, instance={instance}, large={large}"
    logging.info(message)
    output = subset_finding(interaction_values=interaction_values, max_size=max_size)
    assert output is not None
    assert isinstance(output, InteractionValues)
    assert output.index == interaction_values.index
    assert output.max_order == max_size


@pytest.mark.parametrize("max_size", [0, 1, 4, "n_players"])
def test_if_subset_finding_can_be_called(max_size):
    """Test if GaussianCopulaImputer is a subclass of shapiq's Imputer class."""
    game = SOUM(n=10, n_basis_games=50)
    computer = ExactComputer(n_players=game.n_players, game=game)
    interaction_values = computer(index="FSII", order=2)

    max_size = max_size if max_size != "n_players" else game.n_players
    output = subset_finding(interaction_values=interaction_values, max_size=max_size)
    assert output is not None
