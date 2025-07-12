from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from .base import KNNExplainer
from .exceptions import UnsupportedKNNWeightsError

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


class SupportedKNNWeights(Enum):
    """Enumeration of all supported weights types for sklearn KNN models."""

    uniform = "uniform"
    distance = "distance"


class _CommonKNNExplainer(KNNExplainer):
    """Helper class that performs functionality shared between NormalKNNExplainer and WeightedKNNExplainer."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
    ) -> None:
        """Initializes the class.

        This methods extracts the training data as well as the parameter :math:`k` from the provided KNN model and stores them as class members.

        Args:
            model: The KNN model to explain. The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit(X, y)`` must be a 1D vector.
            data: This parameter is currently ignored but may be used in future versions.
            labels: This parameter is currently ignored but may be used in future versions.
            class_index: The class index of the model to explain. Defaults to ``1``.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
            shapiq_student.explainer.knn.exceptions.MultiOutputKNNError: The constructor was called with a model that uses multi-output classification.
        """
        # If this class is instantiated directly, automagically dispatch between normal and weighted KNN according to
        # the given model's configuration

        if self.__class__ is _CommonKNNExplainer:
            explainer_class = get_knn_explainer_class(model)
            self.__class__ = explainer_class
            explainer_class.__init__(
                self,
                model=model,
                class_index=class_index,
            )
            return

        super().__init__(model, class_index=class_index)

        # The type of the superclass's `model` attribute is to broad, since it also allows for other KNN explainers
        # To circumvent this, we store the model separately in an attribute with a narrower type
        self.knn_model = model
        self.k = self.knn_model.n_neighbors  # type: ignore[attr-defined]


def get_knn_explainer_class(
    model: KNeighborsClassifier,
) -> type[_CommonKNNExplainer]:
    """Returns the appropriate subclass of CommonKNNExplainer for the given model."""
    from .normal_knn import NormalKNNExplainer
    from .weighted_knn import WeightedKNNExplainer

    weights = model.weights  # type: ignore[attr-defined]

    if weights == SupportedKNNWeights.uniform.value:
        return NormalKNNExplainer
    if weights == SupportedKNNWeights.distance.value:
        return WeightedKNNExplainer
    raise UnsupportedKNNWeightsError(
        unsupported_weights=weights,
        allowed_weights=[member.value for member in SupportedKNNWeights],
    )
