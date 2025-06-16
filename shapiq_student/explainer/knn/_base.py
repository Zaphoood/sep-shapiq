"""Implementation of KNNExplainerBase class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapiq import Explainer
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


class KNNExplainerBase(Explainer):
    """Base class for all KNN explainers.

    In the constructor, training data and paramater k are extracted from the model.
    """

    def __init__(
        self,
        model: KNeighborsClassifier,
    ) -> None:
        """Initialize the KNNExplainerBase class.

        Args:
            model: The KNN model to explain. Must be an instance of sklearn.neighbors.KNeighborsClassifier.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
        """
        check_is_fitted(model)

        self.model = model
        self.k = self.model.n_neighbors

        self.X_train = model._fit_X  # noqa: SLF001

        y_train_indices = model._y  # noqa: SLF001
        y_train_classes = model.classes_

        # TODO(Zaphoood): Consider disallowing y_train being a matrix
        if y_train_indices.ndim == 1:
            self.y_train = y_train_classes[y_train_indices]
        else:
            n_outputs = y_train_indices.shape[1]
            self.y_train = np.empty((self.X_train.shape[0], n_outputs))

            for col, (y_current_train_classes, y_current_train_indices) in enumerate(
                zip(y_train_classes, y_train_indices.T, strict=False)
            ):
                self.y_train[:, col] = y_current_train_classes[y_current_train_indices]
