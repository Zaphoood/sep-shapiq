"""Custom excpetions for the imputer module."""

from __future__ import annotations


class EmptyDataError(ValueError):
    """Exception raised when the training data is empty."""

    def __init__(self) -> None:
        """Initialize the EmptyDataError exception.

        This exception is raised when the training data provided to an imputer is empty.
        """
        super().__init__("Training data is empty.")
