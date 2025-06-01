# AIML-SEP 2025: shapiq_student - Extending shapiq with additional functionalities

`shapiq_student` is a Python package that extends the functionalities of [`shapiq`](https://github.com/mmschlk/shapiq), a library for explaining machine learning models with Shapley interactions.

## 🛠️ Install and Workflow

`shapiq_student` is intended to work with **Python 3.10 and above**.

## Development

Follow these steps to get started with working on `shapiq_student`

### Install `uv`

Follow [these steps](https://docs.astral.sh/uv/getting-started/installation/) to install the Python package and project manager `uv` on your machine.

### Install project dependencies

Run `uv sync` to install the project's dependencies, including development tools. This will automatically create a virtual environment for the project and install everything there.
To execute any command from within this virtual environment, use `uv run <your command>`. For example, use `uv run python` to get start a Python REPL.

### Set up pre-commit hooks

Pre-commit hooks are a feature of git that allows you to run scripts before every commit, e.g. to make sure your changes adhere to code style regulations.
The tool `pre-commit` helps to install, manage and execute pre-commit hooks. It should already be installed as part of the projects dependencies in the previous step.
To set up `pre-commit` and register it as a git pre-commit hook, run

```sh
uv run pre-commit install
```

Now, every time you run `git commit`, a bunch of scripts defined in `.pre-commit-config.yaml` will be run on your staged changes. If anything is wrong, the command will fail and no commit will be made.
Some checks (such as `ruff`) will also automatically fix some of the problems in your code. These will then appear as unstaged changes in the repository. You now have the option to check the automatic fixes, then stage them and run `git commit` again.

To manually run `pre-commit` on all files in the repository (not just staged changes), run `uv run pre-commit run --all-files`.

## 📜 License

This project is licensed under the [MIT License](https://github.com/mmschlk/shapiq/blob/main/LICENSE).

---

Built with ❤️ by the shapiq_student team.
