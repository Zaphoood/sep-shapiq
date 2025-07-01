"""Custom excpetions for the imputer module."""

from __future__ import annotations


class EmptyDataError(ValueError):
    """Exception raised when the training data is empty."""

    def __init__(self) -> None:
        """Initialize the EmptyDataError exception.

        This exception is raised when the training data provided to an imputer is empty.
        """
        super().__init__("Training data is empty.")


class CategoricalFeatureError(ValueError):
    """Exception raised when categorical features are detected."""

    def __init__(self, feature_indices: list[int]) -> None:
        """Initialize the error with the categorical feature indices.

        Parameters
        ----------
        feature_indices : list[int]
            List of indices of features that are categorical
        """
        # Convert indices to f1, f2, ...
        feature_names = [f"f{i + 1}" for i in feature_indices]
        message = (
            f"The following are categorical features: {', '.join(feature_names)}. "
            "Gaussian approach does not support categorical features."
        )
        super().__init__(message)
        self.feature_indices = feature_indices
