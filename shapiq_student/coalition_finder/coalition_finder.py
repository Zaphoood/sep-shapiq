"""Implementation of Coalition Finding Algorithm."""

from __future__ import annotations

from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues


def get_coalitions(max_size: int) -> np.ndarray:
    """Returns all 2^n_players coalitions as a binary matrix (shape=(2^n, n_players)).

    Each row is a 0/1 vector of length n_players.
    """
    all_coals = list(product([0, 1], repeat=max_size))
    return np.array(all_coals, dtype=int)


def get_explanation_one_feature(
    coalition: np.ndarray, interaction_values: InteractionValues
) -> float:
    """Sums all e_i for individual players i ∈ S (|T|=1)."""
    total = 0.0
    # Finde alle Einträge i, für die coalition[i]==1
    for (subset,), idx in interaction_values.interaction_lookup.items():
        if coalition[subset] == 1:
            total += interaction_values.values[idx]
    return total


def get_explanation_two_features(
    coalition: np.ndarray, interaction_values: InteractionValues
) -> float:
    """Sums all e_{i,j} for pairs {i,j} ⊆ S (|T|=2)."""
    total = 0.0
    # Erzeuge alle 2er-Untermengen der Indizes, die in dieser Koalition aktiv sind
    idx_value_equals_one = np.nonzero(coalition)[0]
    for i, j in combinations(idx_value_equals_one, 2):  # Erzeugt alle 2er-combis
        idx = interaction_values.interaction_lookup.get((i, j))
        # Da Lookup vielleicht nur geordnete Tupel speichert, teste auch (j,i) - tbh glaub können wir rausnehmen?
        if idx is None:
            idx = interaction_values.interaction_lookup.get((j, i))
        if idx is not None:
            total += interaction_values.values[idx]
    return total


def get_explanation_more_features(
    coalition: np.ndarray, interaction_values: InteractionValues, max_order: int
) -> float:
    """Sums all e_T for T ⊆ S with 3 ≤ |T| ≤ max_order."""
    total = 0.0
    idx_value_equals_one = np.nonzero(coalition)[0]
    # Für jede Ordnung k = 3..max_order
    for k in range(3, max_order + 1):
        for subset in combinations(idx_value_equals_one, k):
            idx = interaction_values.interaction_lookup.get(subset)
            if idx is not None:
                total += interaction_values.values[idx]
    return total


def subset_finding(
    interaction_values: InteractionValues, coalition_size: int, max_order: int | None = None
) -> tuple[np.ndarray, float]:
    r"""Searches all S ⊆ N with |S|=coalition_size.

    and returns the coalition that maximizes \hat v_e(S),
    along with its value.
    """
    # Bestimme Anzahl der Player - but not sure if even needed
    n_players = interaction_values.n_players
    if max_order is None:
        max_order = n_players

    # Baue die Baseline e0 (falls in interaction_lookup vorhanden)
    e0 = 0.0
    if () in interaction_values.interaction_lookup:
        idx0 = interaction_values.interaction_lookup[()]
        e0 = interaction_values.values[idx0]

    best_val = -np.inf
    best_coal: np.ndarray | None = None

    # Alle Koalitionen erzeugen und filtern
    for coalition in get_coalitions(n_players):
        if coalition.sum() != coalition_size:
            continue

        # 1) Einzelne Player
        v1 = get_explanation_one_feature(coalition, interaction_values)
        # 2) Zweier-Interaktionen
        v2 = get_explanation_two_features(coalition, interaction_values)
        # 3) Höhere Ordnungen
        v3 = get_explanation_more_features(coalition, interaction_values, max_order)

        total = e0 + v1 + v2 + v3
        if total > best_val:
            best_val = total
            best_coal = coalition.copy()

    if best_coal is None:
        error_msg = "keine Koalition gefunden!"
        raise ValueError(error_msg)
    return best_coal, best_val
