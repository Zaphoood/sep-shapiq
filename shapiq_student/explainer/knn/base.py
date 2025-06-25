"""Implementation of ``KNNExplainerBase`` class and associated utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapiq import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier


class KNNExplainerBase(Explainer):
    """Base class for KNN explainers."""

    model: KNeighborsClassifier
    """The KNN model provided in the constructor."""

    X_train: np.ndarray
    """Training data features extracted from the model."""

    y_train: np.ndarray
    """Training data labels extracted from the model. This array simply resolves the indirection of looking up class indices from ``y_train_indices`` in ``y_train_classes``."""

    k: int
    """The parameter ``k`` of the model."""

    def __init__(
        self,
        model: KNeighborsClassifier,
    ) -> None:
        """Initializes the KNNExplainerBase class.

        This methods extracts the training data as well as the parameter ``k`` from the provided KNN model and stores it as class attributes.

        Args:
            model: The KNN model to explain. Must be an instance of ``sklearn.neighbors.KNeighborsClassifier``.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
        """
        check_is_fitted(model)

        self.model = model
        self.k = self.model.n_neighbors

        self.X_train = model._fit_X  # type: ignore[attr-defined] # noqa: SLF001

        self.y_train_indices = model._y  # type: ignore[attr-defined] # noqa: SLF001
        self.y_train_classes = model.classes_

        # TODO(Zaphoood): Consider disallowing y_train being a matrix
        if self.y_train_indices.ndim == 1:
            self.y_train = self.y_train_classes[self.y_train_indices]
        else:
            n_outputs = self.y_train_indices.shape[1]
            self.y_train = np.empty((self.X_train.shape[0], n_outputs))

            for col, (y_current_train_classes, y_current_train_indices) in enumerate(
                zip(self.y_train_classes, self.y_train_indices.T, strict=False)
            ):
                self.y_train[:, col] = y_current_train_classes[y_current_train_indices]


def interaction_lookup_from_knn_shapley_values(
    shapley_values: npt.NDArray[np.floating],
) -> InteractionValues:
    """Convert an array of Shapley Values to a ``shapiq.interaction_values.InteractionValues`` object.

    Args:
        shapley_values: A ``np.ndarray`` containing the Shapley Value of the ith training point at index i.

    Returns:
        An ``InteractionValues`` object containing the provided Shapley Values with an appropriate ``interaction_lookup`` dict and with ``min_order == max_order == 1`` set.
    """
    n_players = shapley_values.shape[0]
    interaction_lookup: dict[tuple[int, ...], int] = {(i,): i for i in range(n_players)}

    return InteractionValues(
        shapley_values,
        "SV",
        min_order=1,
        max_order=1,
        n_players=n_players,
        baseline_value=0,
        interaction_lookup=interaction_lookup,
    )
