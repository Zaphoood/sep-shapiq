"""Tests the WKNN Explainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn.wknn import WKNNExplainer

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


@dataclass
class WKNNTestCase:
    """Defines a test case for a WKNN classifier."""

    X_train: npt.NDArray[np.floating]
    y_train: npt.NDArray[np.floating]
    class_index: int
    x_val: npt.NDArray[np.floating]
    sv_expected: None | npt.NDArray[np.floating]
    k: int
    n_bits: int


def test_wknn_hardcoded_example():
    """Tests that the Shapley Values computed by WKNNExplainer are correct for a hard-coded example."""
    test_cases = [
        WKNNTestCase(
            X_train=np.array([[-8, 0], [-1.5, 0], [-0.5, 0], [0.5, 0], [1.5, 0]]),
            y_train=np.array([0, 0, 0, 1, 1]),
            class_index=0,
            x_val=np.array([[0.2, 0]]),
            sv_expected=None,
            k=3,
            n_bits=5,
        )
    ]

    for test_case in test_cases:
        model = KNeighborsClassifier(n_neighbors=test_case.k, weights="distance")
        model.fit(test_case.X_train, test_case.y_train)

        explainer = WKNNExplainer(model, class_index=test_case.class_index, n_bits=test_case.n_bits)
        iv_actual = explainer.explain(test_case.x_val)
        sv_actual = _get_ordered_values(iv_actual)

        # TODO(Zaphoood): Remove this option once there is real test data
        if test_case.sv_expected is None:
            print(  # noqa: T201
                "No expected SV defined for test case; will skip value comparison and perform only sanity checks."
            )
            assert sv_actual.shape[0] == test_case.X_train.shape[0]
        else:
            assert np.allclose(sv_actual, test_case.sv_expected)


def _get_ordered_values(shapley_values: InteractionValues) -> npt.NDArray[np.floating]:
    """Return shapley values ordered by player index.

    Args:
        shapley_values: An InteractionValues object with max_order==1

    Returns:
        An np.ndarray of shape (n_players,) containing at index i the Shapley value of player i.
    """
    if shapley_values.max_order != 1:
        msg = f"Max order must be 1 but was {shapley_values.max_order}"
        raise ValueError(msg)

    out = np.zeros((shapley_values.n_players,))

    for coalition, lookup_idx in shapley_values.interaction_lookup.items():
        if coalition == ():
            continue
        out[coalition[0]] = shapley_values.values[lookup_idx]

    return out


if __name__ == "__main__":
    test_wknn_hardcoded_example()
