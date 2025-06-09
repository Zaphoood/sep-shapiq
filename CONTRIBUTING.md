# 🛠️ Development Guide

Follow the instructions below when working on `shapiq_student`.

## Setup

### Installation and environment

Before starting, please note that `shapiq_student` is intended to work with **Python 3.10 and above**. Follow [these steps](https://docs.astral.sh/uv/getting-started/installation/) to install the Python package and project manager `uv` on your machine.

Run `uv sync` to install the project's dependencies, including development tools. This will automatically create a virtual environment for the project and install everything there.
To execute any command from within this virtual environment, use `uv run <your_command>`. For example, use `uv run python` to start a Python REPL.

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

## Managing project dependencies

To add a new dependency that will be required by the library at runtime, execute `uv add <package_name>`. Dependencies can also be added by manually editing the `pyproject.toml` file, followed by running `uv sync`.
Both ways will modify files `pyproject.toml` and `uv.lock`. Make sure to commit them afterwards!

If you want to add a dependency that is only needed at development time and not required for the library to function, specify the dependency group `dev` with the argument `--group dev`: For example, to add the `mypy` type checker, you would run `uv add mypy --group dev`. However, the development dependency group is also further subdivided in groups `test`, `lint` etc. (defined in `pyproject.toml`); try to use the most appropriate group for the dependency you're adding. This way, no unnecessary dependencies are installed in the GitHub Actions workflows that only require a subset of the development tools.

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

After every push to the `main` branch, an updated version of the documentation will be generated and published to [GitHub Pages](https://zaphoood.github.io/sep-shapiq).
