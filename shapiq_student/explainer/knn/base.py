"""Implementation of ``KNNExplainerBase`` class and associated utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from shapiq import Explainer
from shapiq.interaction_values import InteractionValues
from sklearn.utils.validation import check_is_fitted

from shapiq_student.explainer.knn.exceptions import MultiOutputKNNError

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

    def __init__(self, model: KNeighborsClassifier, class_index: int) -> None:
        """Initializes the class.

        This methods extracts the training data from the provided KNN or TNN model and stores it in class attributes.

        Args:
            model: The KNN model to explain. Must be an instance of ``sklearn.neighbors.KNeighborsClassifier``.
                The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit()`` must be a 1D vector.

            class_index: The class index of the model to explain. Note that, as opposed to the parent class ``shapiq.explainer.Explainer``, the parameter is not optional here.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.

            shapiq_student.explainer.knn.exceptions.MultiOutputKNNError: The constructor was called with a model that uses multi-output classification.
        """
        check_is_fitted(model)

        super().__init__(model, data=None, class_index=class_index, index="SV", max_order=1)

        self.model = model
        self.k = self.model.n_neighbors  # type: ignore[attr-defined]

        self.X_train = model._fit_X  # type: ignore[attr-defined] # noqa: SLF001

        self.y_train_indices = cast("npt.NDArray[np.integer]", model._y)  # type: ignore[attr-defined] # noqa: SLF001
        self.y_train_classes = cast("npt.NDArray[np.object_]", model.classes_)

        if self.y_train_indices.ndim != 1:
            raise MultiOutputKNNError

        self.y_train = self.y_train_classes[self.y_train_indices]

        self.class_index = class_index


def interaction_values_from_array(
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


def interaction_values_to_array(
    interaction_values: InteractionValues,
) -> npt.NDArray[np.floating]:
    """Extract an array of Shapley Values from a ``shapiq.interaction_values.InteractionValues`` object.

    Args:
        interaction_values: An InteractionValues object with ``max_order==1``

    Returns:
        An ``np.ndarray`` of shape ``(n_players,)`` containing at index i the Shapley value of player i.
    """
    if interaction_values.max_order != 1:
        msg = f"Max order must be 1 but was {interaction_values.max_order}"
        raise ValueError(msg)

    out = np.zeros((interaction_values.n_players,))

    for coalition, lookup_idx in interaction_values.interaction_lookup.items():
        if coalition == ():
            continue
        out[coalition[0]] = interaction_values.values[lookup_idx]

    return out
