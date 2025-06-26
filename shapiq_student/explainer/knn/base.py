"""Implementation of ``KNNExplainerBase`` class and associated utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from shapiq import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier


class KNNExplainerBase(Explainer):
    """Base class for KNN explainers."""

    # The model attribute of the Explainer is defined in a non-optimal way,
    # using a type variable `Model`, which has no meaning. This is why we need
    # to supress a type error here.
    model: KNeighborsClassifier  # type: ignore[assignment]
    """The KNN model provided in the constructor."""

    X_train: npt.NDArray[np.floating]
    """Training data features extracted from the model."""

    y_train_indices: npt.NDArray[np.integer]
    """Training data labels as indices into the classes array."""

    y_train_classes: npt.NDArray[np.object_]
    """Classes that appear in the model's training data."""

    y_train: npt.NDArray[np.object_]
    """Training data labels extracted from the model. This array simply resolves the indirection of looking up class indices from ``y_train_indices`` in ``y_train_classes``."""

    k: int
    """The parameter ``k`` of the model."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
    ) -> None:
        """Initializes the KNNExplainerBase class.

        This methods extracts the training data as well as the parameter ``k`` from the provided KNN model and stores it as class attributes.

        Args:
            model: The KNN model to explain. Must be an instance of ``sklearn.neighbors.KNeighborsClassifier``.
            class_index: The class index of the model to explain. For more information, see the API of the `base class <https://shapiq.readthedocs.io/en/latest/api/shapiq.explainer.base.html#shapiq.explainer.base.Explainer>`_.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
        """
        check_is_fitted(model)

        super().__init__(model, data=None, class_index=class_index, index="SV", max_order=1)

        self.model = model
        self.k = self.model.n_neighbors  # type: ignore[attr-defined]

        self.X_train = model._fit_X  # type: ignore[attr-defined] # noqa: SLF001

        self.y_train_indices = cast("npt.NDArray[np.integer]", model._y)  # type: ignore[attr-defined] # noqa: SLF001
        self.y_train_classes = cast("npt.NDArray[np.object_]", model.classes_)

        # TODO(Zaphoood): Consider disallowing y_train being a matrix
        if self.y_train_indices.ndim == 1:
            self.y_train = self.y_train_classes[self.y_train_indices]
        else:
            n_outputs = self.y_train_indices.shape[1]
            self.y_train = np.empty((self.X_train.shape[0], n_outputs), dtype=np.object_)

            for col, (current_y_train_classes, current_y_train_indices) in enumerate(
                zip(self.y_train_classes, self.y_train_indices.T, strict=False)
            ):
                self.y_train[:, col] = current_y_train_classes[current_y_train_indices]


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
