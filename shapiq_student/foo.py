"""A sample module that needs a docstring in order for the linter to be happy."""

from __future__ import annotations


class Foo:
    """A sample class to illustrate the project structure and DevOps setup."""

    def bar(self, x: int, y: int) -> int:
        """Adds two integers and returns the result.

        Args:
            x (int): The first integer to add.
            y (int): The second integer to add.

        Returns:
            int: The sum of `x` and `y`.
        """
        return x + y


def baz() -> None:
    """A top-level function that prints 'Hello, World!'."""
    print("Hello, world")  # noqa: T201
