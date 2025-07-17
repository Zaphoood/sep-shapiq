"""Example test case for the Explainer class."""

from __future__ import annotations

import numpy as np
from shapiq.games.imputer.base import Imputer

from shapiq_student import GaussianCopulaImputer, GaussianImputer

from .utils import get_random_coalitions


def dummy_model(x: np.ndarray) -> np.ndarray:
    """A dummy model that returns the sum of the features.

    Note:
        This callable is just here that we satisfy the Imputer's model parameter and that we can
        check if the Imputer can be called with coalitions and returns a vector of "predictions".

    Args:
        x: Input data as a 2D numpy array with shape (n_samples, n_features).

    Returns:
        A 1D numpy array with the sum of the features for each sample.
    """
    return np.sum(x, axis=1)


class TestImputers:
    """Tests for calling the GaussianImputer and GaussianCopulaImputer with coalitions."""

    @staticmethod
    def get_coalitions(data_test) -> np.ndarray:
        """Generate random coalitions for testing."""
        X, _ = data_test
        n_features = X.shape[1]
        # get random coalitions (intentionally without specifying random seed set)
        coalitions = get_random_coalitions(n_features=n_features, n_coalitions=10)
        assert coalitions.shape == (10, n_features), "Coalitions should have the correct shape."
        return coalitions

    def test_gaussian_copula_imputer_init(self, data_test, x_explain):
        """Test init of GaussianCopulaImputer."""
        assert issubclass(GaussianCopulaImputer, Imputer), (
            "GaussianCopulaImputer should be a subclass of Imputer."
        )

        x_test, _ = data_test
        n_features = x_test.shape[1]
        imputer = GaussianCopulaImputer(model=dummy_model, data=x_test)
        assert isinstance(imputer, GaussianCopulaImputer)
        assert isinstance(imputer, Imputer)
        assert imputer.x is None, "x should be None initially."
        assert imputer.n_players == n_features

        # test with x set to x_explain
        imputer = GaussianCopulaImputer(model=dummy_model, data=x_test, x=x_explain)
        assert isinstance(imputer, GaussianCopulaImputer)
        assert isinstance(imputer, Imputer)
        assert np.array_equal(imputer.x, x_explain), "x should be set to x_explain."

    def test_gaussian_imputer_init(self, data_test, x_explain):
        """Test init of GaussianImputer."""
        assert issubclass(GaussianImputer, Imputer), (
            "GaussianCopulaImputer should be a subclass of Imputer."
        )

        x_test, _ = data_test
        n_features = x_test.shape[1]
        imputer = GaussianImputer(model=dummy_model, data=x_test)
        assert isinstance(imputer, GaussianImputer)
        assert isinstance(imputer, Imputer)
        assert imputer.x is None, "x should be None initially."
        assert imputer.n_players == n_features

        # test with x set to x_explain
        imputer = GaussianImputer(model=dummy_model, data=x_test, x=x_explain)
        assert isinstance(imputer, GaussianImputer)
        assert isinstance(imputer, Imputer)
        assert np.array_equal(imputer.x, x_explain), "x should be set to x_explain."

    def test_gaussian_can_be_called(self, data_test, x_explain):
        """Test if GaussianImputer can be called."""
        coalitions = self.get_coalitions(data_test)

        # check the GaussianImputer can be called with coalitions
        x_test, _ = data_test
        imputer = GaussianImputer(model=dummy_model, data=x_test)
        imputer.fit(x=x_explain)
        output = imputer(coalitions=coalitions)
        assert isinstance(output, np.ndarray)
        assert output.shape == (10,), "Output should be a vector of predictions."

    def test_gaussian_copula_can_be_called(self, data_test, x_explain):
        """Test if GaussianCopulaImputer can be called."""
        coalitions = self.get_coalitions(data_test)

        # check the GaussianCopulaImputer can be called with coalitions
        x_test, _ = data_test
        imputer = GaussianCopulaImputer(model=dummy_model, data=x_test)
        imputer.fit(x=x_explain)
        output = imputer(coalitions=coalitions)
        assert isinstance(output, np.ndarray)
        assert output.shape == (10,), "Output should be a vector of predictions."
