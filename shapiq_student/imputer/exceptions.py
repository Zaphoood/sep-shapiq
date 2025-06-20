"""Custom excpetions for the imputer module."""

from __future__ import annotations


class FeatureNamesLengthError(ValueError):
    """Exception raised when feature names length doesn't match number of features."""

    def __init__(self, feature_names_length: int, n_features: int) -> None:
        """Initialize the error with the mismatched lengths.

        Parameters
        ----------
        feature_names_length : int
            Length of the provided feature names list
        n_features : int
            Number of features in the data
        """
        self.feature_names_length = feature_names_length
        self.n_features = n_features
        message = f"feature_names length {feature_names_length} != number of features {n_features}"
        super().__init__(message)
