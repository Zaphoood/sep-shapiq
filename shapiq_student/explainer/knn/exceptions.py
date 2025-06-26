"""Defines custom exceptions used by KNN Explainers."""

from __future__ import annotations


class MultiOutputKNNClassifierError(Exception):
    """Exception that is raised when a user tries to initialize a KNN Explainer with a model that has multiple output columns."""

    def __init__(self) -> None:
        """Initializes the exception object."""
        super().__init__(
            "Multi-output KNN classifiers are not supported. Make sure to pass the training labels as a 1D vector when calling `model.fit()`."
        )
