"""Tests for module foo."""

from __future__ import annotations

from shapiq_student.foo import Foo, baz


def test_bar() -> None:
    """Tests that the bar method performs addition."""
    foo = Foo()
    assert foo.bar(2, 3) == 5  # noqa: PLR2004


def test_baz() -> None:
    """Tests that the baz function can be called."""
    baz()
