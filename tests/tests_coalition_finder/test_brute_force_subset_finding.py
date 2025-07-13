"""Tests brute-force implementation of subset-finding."""

from __future__ import annotations

import numpy as np
from shapiq import InteractionValues

from shapiq_student.coalition_finder import brute_force_subset_finding


def test_brute_force():
    """Tests on handcrafted example."""
    # Example data for 3 players (0,1,2)
    values = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )  # baseline, 3 shapley values, 3 shapley interaction values
    interaction_lookup = {
        (): 0,
        (0,): 1,
        (1,): 2,
        (2,): 3,
        (0, 1): 4,
        (0, 2): 5,
        (1, 2): 6,
    }

    iv = InteractionValues(
        values=values,
        interaction_lookup=interaction_lookup,
        n_players=3,
        min_order=1,
        max_order=2,
        baseline_value=0.0,
        index="SII",
    )

    # Find min/max coalitions of size 2
    result_iv = brute_force_subset_finding(iv, coalition_size=2)

    # p rint("Resulting values:", result_iv.values)
    # p rint("Resulting lookup:", result_iv.interaction_lookup)
    # Should print: values [0.4 0.6], lookup {(0, 1): 0, (1, 2): 1}
    # TODO(murscht): delete print statements later/keep now to remember faster how a dict works

    assert np.isclose(result_iv.values[0], 0.4)
    assert np.isclose(result_iv.values[1], 0.6)
    assert result_iv.interaction_lookup[(0, 1)] == 0
    assert result_iv.interaction_lookup[(1, 2)] == 1
