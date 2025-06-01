"""Tests for module foo."""

from __future__ import annotations

from shapiq_student.foo import Foo


def test_bar():
    """Tests that the bar method performs addition."""
    foo = Foo()
    assert foo.bar(2, 3) == 5  # noqa: PLR2004
