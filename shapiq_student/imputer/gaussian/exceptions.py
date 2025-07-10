"""Custom exceptions for the imputer module."""

from __future__ import annotations


class CategoricalFeatureError(ValueError):
    """Exception raised when categorical features are detected."""

    def __init__(self, feature_indices: list[int]) -> None:
        """Initializes the error with the categorical feature indices.

        Args:
            feature_indices (list[int]): List of indices of features that are categorical.
        """
        # Convert indices to f1, f2, ...
        feature_names = [f"f{i + 1}" for i in feature_indices]
        message = (
            f"The following are categorical features: {', '.join(feature_names)}. "
            "Gaussian approach does not support categorical features."
        )
        super().__init__(message)
        self.feature_indices = feature_indices
