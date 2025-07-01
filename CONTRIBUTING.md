# 🛠️ Development Guide

Follow the instructions below when working on `shapiq_student`.

## Setup

### Installation and environment

Before starting, please note that `shapiq_student` is intended to work with **Python 3.10 and above**. Follow [these steps](https://docs.astral.sh/uv/getting-started/installation/) to install the Python package and project manager `uv` on your machine.

Run `uv sync` to install the project's dependencies, including development tools. This will automatically create a virtual environment for the project and install everything there.
To execute any command from within this virtual environment, use `uv run <your_command>`. For example, use `uv run python` to start a Python REPL.

Before you start commiting any changes, run `uv run pre-commit install` to enable pre-commit hooks. For an explanation on what pre-commit hooks are, see [below](#pre-commit-hooks).

## Code quality checks

### Pre-commit hooks

Pre-commit hooks are a feature of git that allows you to run scripts before every commit, e.g. to make sure your changes adhere to code style regulations.
The tool `pre-commit` helps to install, manage and execute pre-commit hooks. It should already be installed as part of the project's dependencies from the previous step.
To set up `pre-commit` and register it as a git pre-commit hook, run

```sh
uv run pre-commit install
```

Now, every time you run `git commit`, a bunch of scripts defined in `.pre-commit-config.yaml` will be run on your staged changes. If anything is wrong, the command will report the problem and the commit will be aborted.
Some checks (such as `ruff`) will also automatically fix some of the problems in your code. These will then appear as unstaged changes in the repository. You now have the option to check the automatic fixes, then stage them and run `git commit` again.

To manually run `pre-commit` on all files in the repository (not just staged changes), run `uv run pre-commit run --all-files`.

### Linting with `ruff`

This project uses [`ruff`](https://docs.astral.sh/ruff/) for linting. When you run `ruff` on a file that has problems, it will provide an error code and message for each one (such as `D104 Missing docstring in public package`). Paste the error code into the search box at the `ruff` [homepage](https://docs.astral.sh/ruff/) to get an in-depth explanation (like [this one](https://docs.astral.sh/ruff/rules/undocumented-public-package/)) on why the code in question is problematic and what to do instead.

Linter errors can be very strict, so sometimes it makes sense to surpress (disable) them. However, you should do this with caution. Always try fixing the linter error first rather than surpressing it.

There are generally three ways to disable linter errors:

#### In-line

If you wish to disable an error only for a specific line, add `  # noqa: <error code>` at the end of the line in question.
For example, you may need to use an import statement that's not at the top of the file, since you need to set an environment variable first to configure a library import. In this case it's perfectly fine to surpress `E402 Module level import not at top of file`:

```python
import os
os.environ["MY_SETTING"] = "value"
import my_library  # noqa: E402
```

#### Per file

If you wish to disable an error for a specific file or a set of files, add a corresponding entry to the section `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`.
For example, we often want to re-export classes or functions in `__init__.py` files, since this makes writing import statements less cumbersome for the library users. This would
trigger an error ```F401 `SomeClass` imported but unused```, wich we can surpress like so:

```toml
[tool.ruff.lint.per-file-ignores]
"__init__.py" = [
    "F401",  # unused imports are okay here
    ...
]
```

Ideally, add a comment explaining why it is okay to surpress this kind of error in this kind of file, or at least paste the explanation of the error code. For more info, see the `ruff` docs for [per-file ignores](https://docs.astral.sh/ruff/settings/#lint_per-file-ignores).

#### Everywhere

To disable a specific error code everywhere, add it to the array `ignore` in the section `[tool.ruff.lint]` in `pyproject.toml`. Use this option with care and only if you are completely sure that this kind of error should never be flagged by `ruff`! Always include a justification or the explanation of the error code as a comment:

```toml
[tool.ruff.lint]
ignore = [
    "D203",  # conflicts with D211
    ...
]
```

The example above illustrates a situation where it is useful to surpress a rule completely: The rules `D203` and `D211` are in conflict with one another, since the first requires a blank line before a class docstring, while the second requires there to be _no_ blank line. Therefore, either the first or the second rule will always trigger an error, unless one of them is disabled.

Note that in our case, the `D203` rule is automatically disabled by setting the `google` style convention (see section `[tool.ruff.lint.pydocstyle]` of `pyproject.toml`).

For more info, see the `ruff` docs for [ignore](https://docs.astral.sh/ruff/settings/#lint_ignore).

## Managing project dependencies

To add a new dependency that will be required by the library at runtime, execute `uv add <package_name>`. Dependencies can also be added by manually editing the `pyproject.toml` file, followed by running `uv sync`.
Both ways will modify files `pyproject.toml` and `uv.lock`. Make sure to commit them afterwards!

If you want to add a dependency that is only needed at development time and not required for the library to function, specify the dependency group `dev` with the argument `--group dev`: For example, to add the `mypy` type checker, you would run `uv add mypy --group dev`. However, the development dependency group is also further subdivided in groups `test`, `lint` etc. (defined in `pyproject.toml`); try to use the most appropriate group for the dependency you're adding. This way, no unnecessary dependencies are installed in the GitHub Actions workflows that only require a subset of the development tools.

## Type checking

This project uses [`mypy`](https://mypy.readthedocs.io/en/stable/) for static type checking. In Python, type annotations (or type hints) can be added to code, which do not have _any impact_ on runtime behavior, but catching type-related errors by static code analysis. An in-depth article on type systems and types in Python can be found [here](https://realpython.com/python-type-checking/). This [cheatsheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) is very useful for getting started with writing type annotations.

The `mypy` type checker is _not_ configured as a pre-commit hook (for now). Reasons are that it can be quite slow and is annoying to configure. But there is a GitHub Actions workflow that runs the type checker on every PR.

You should run the type checker regularly, and especially before opening a PR, by executing:

```sh
uv run mypy -p shapiq_student
```

This tells `mypy` to type checking the entire `shapiq_student` package (hence the `-p` flag). Be warned that this will take several seconds when you run it for the first time; don't despair, subsequent runs will be much faster thanks to caching.

## Tests

When adding new functionalities, write corresponding unittests and add them to the directory `tests/`. Our testing framework is [pytest](https://docs.pytest.org/en/stable/getting-started.html). [Here](https://docs.pytest.org/en/stable/how-to/index.html) you can find some instructions on writing tests and using the `pytest` command.

Our goal is to maintain a code coverage of at least 92%. You can use the following command to generate a nice code coverage report, which you can open in your browser, allowing you to inspect each file for coverage:

```sh
uv run pytest --cov=shapiq_student --cov-report=html
```

This will generate a directory `htmlcov`. Open `htmlcov/index.html` in your browser to view the coverage report.

**Note:** PRs which move the code coverage percentage below the minimum threshold will be blocked for merging; in this case, add unittests for untested code to satisfy the threshold.

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/). The documentation source files are stored at [`docs/source/`](docs/source/) and are written in the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) syntax. A good starting place is `docs/source/index.rst`.

Furthermore, an automatic API reference of the library will be created from the [docstrings](https://peps.python.org/pep-0257/#what-is-a-docstring) in the Python code.
When adding new modules, classes, functions or methods, include a docstring, and update docstrings when making changes to existing code. Make sure to follow [Google's docstring conventions](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

**Important**: When you add a new top-level package (e.g. `shapiq_student.explainer`), you must add it to `docs/source/api.rst` in the 'autosummary' section, analogous to the existing entries.

After every push to the `main` branch, an updated version of the documentation will be generated and published to [GitHub Pages](https://zaphoood.github.io/sep-shapiq).

### Generating locally

To generate the documentation locally, run

```sh
uv run sphinx-build -M html docs/source docs/build --fail-on-warning
```

This will save the generated output to `docs/build`. Open `docs/build/html/index.html` in your browser to view the documentation. It is recommended to do this after writing docstrings or new documentation, in order to make sure that Sphinx generated what you intended.

If you encounter any problems, deleting `docs/build` and `docs/source/api` may help. The latter is generated by Sphinx when discovering modules to include in the API reference, and may include stale references.
