"""Example test case for the Explainer class."""

from __future__ import annotations

from shapiq.games.imputer.base import Imputer

from shapiq_student import GaussianCopulaImputer, GaussianImputer


def test_is_gaussian_imputer_an_imputer_class():
    """Test if GaussianImputer is a subclass of shapiq's Imputer class."""
    assert issubclass(GaussianImputer, Imputer), "GaussianImputer should be a subclass of Imputer."


def test_is_gaussian_copula_imputer_an_imputer_class():
    """Test if GaussianCopulaImputer is a subclass of shapiq's Imputer class."""
    assert issubclass(GaussianCopulaImputer, Imputer), (
        "GaussianCopulaImputer should be a subclass of Imputer."
    )
