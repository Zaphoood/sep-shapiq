"""Provides a utility function for dynamically selecting the right explainer for a KNN model."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from .exceptions import UnsupportedKNNWeightsError

if TYPE_CHECKING:
    from shapiq_student.explainer.knn.base import KNNExplainer

    from .custom_types import KNNClassifierModel


class SupportedKNNWeights(Enum):
    """Enumeration of all supported weights types for sklearn KNN models."""

    uniform = "uniform"
    distance = "distance"


class KNNExplainerMode(Enum):
    """Enumeration of all modes in which a KNN explainer can operate."""

    basic = "normal"
    weighted = "weighted"
    threshold = "threshold"


def get_explainer_class_and_mode(
    model: KNNClassifierModel,
) -> tuple[KNNExplainerMode, type[KNNExplainer]]:
    """Returns the appropriate subclass of KNNExplainer for the given model."""
    from .knn import BasicKNNExplainer
    from .tknn import TKNNExplainer
    from .wknn import WKNNExplainer

    if isinstance(model, KNeighborsClassifier):
        weights = model.weights  # type: ignore[attr-defined]

        if weights == SupportedKNNWeights.uniform.value:
            return KNNExplainerMode.basic, BasicKNNExplainer
        if weights == SupportedKNNWeights.distance.value:
            return KNNExplainerMode.weighted, WKNNExplainer
        raise UnsupportedKNNWeightsError(
            unsupported_weights=weights,
            allowed_weights=[member.value for member in SupportedKNNWeights],
        )
    if isinstance(model, RadiusNeighborsClassifier):
        return KNNExplainerMode.threshold, TKNNExplainer

    msg = f"Exhaustive check of KNN classifier models in _get_subclass_for_model: Unhandled {model}"
    raise RuntimeError(msg)
