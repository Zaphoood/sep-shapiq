"""Test that the shapiq dependency is installed with at least version 1.3.0."""

from __future__ import annotations

import importlib.metadata

from shapiq import __version__ as shapiq_version


def test_shapiq_version():
    """Test that shapiq is installed with at least version 1.3.0."""
    required_version = "1.3.0"
    assert importlib.metadata.version("shapiq") >= required_version, (
        f"shapiq version {required_version} or higher is required, but found {shapiq_version}."
    )
