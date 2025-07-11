"""Stress tests for Explainers.

Tests the performance of the Explainers on large datasets.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import TYPE_CHECKING, TypedDict

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from shapiq_student import KNNExplainer

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt


def explain_timed(explainer: KNNExplainer, x_explain: npt.NDArray[np.floating]) -> float:
    """Measure the time required to time the given datapoint with the given explainer."""
    t_start = time.time()
    explainer.explain(x_explain)
    t_end = time.time()

    return t_end - t_start


Model = KNeighborsClassifier | RadiusNeighborsClassifier

if TYPE_CHECKING:
    # A function that accepts training data and returns a model trained on that data
    ModelFactory = Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.object_ | np.number]],
        Model,
    ]
    # A function that accepts a trained model an returns a KNN explainer of that model
    ExplainerFactory = Callable[[Model], KNNExplainer]


def explain_timed_many_sizes(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.object_ | np.number],
    x_explain: npt.NDArray[np.floating],
    train_sizes: list[int],
    fit_model: ModelFactory,
    get_explainer: ExplainerFactory,
    *,
    verbose: bool = False,
) -> dict[int, float]:
    """Measure the time required for explaining for multiple different training datasets."""
    timings: dict[int, float] = {}

    for train_size in train_sizes:
        if verbose:
            print(f"{train_size=}")
        if train_size > X_train.shape[0]:
            print(
                f"WARNING: Training set size limit ({train_size}) is larger than actual training set ({X_train.shape[0]})"
            )
        X_train_current = X_train[:train_size]
        y_train_current = y_train[:train_size]
        model = fit_model(X_train_current, y_train_current)
        explainer = get_explainer(model)

        timing = explain_timed(explainer, x_explain)
        if verbose:
            print(f"{timing:.3f}s")
        timings[train_size] = timing

    return timings


def get_normal_knn_factory(k: int) -> ModelFactory:
    """Returns a 'model factory', i. e. a function that will train a normal KNN classifier on some training data."""

    def fit_knn_model(
        X_train: npt.NDArray[np.floating], y_train: npt.NDArray[np.object_ | np.number]
    ) -> KNeighborsClassifier:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        return model

    return fit_knn_model


class StressTestConfig(TypedDict):
    """Configuration of a stress test for an explainer."""

    name: str
    train_sizes: list[int]
    fit_model: ModelFactory
    get_explainer: ExplainerFactory


def save_timings_as_csv(timings: dict[int, float], path: str) -> None:
    """Saves the given timings in CSV format at the given path."""
    with Path(path).open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["train_size", "timing"])
        for key, value in timings.items():
            writer.writerow([key, value])


def stress_test_model(config: StressTestConfig, *, save_as_csv: bool = False) -> None:
    """Run a stress test with the given configuration.

    This will call the explainer on a single explanation point, each time for a model trained on a differently sized dataset.
    """
    print(f"Running stress test: {config['name']}")

    max_train_size = config["train_sizes"][-1]
    n_samples = max_train_size + 1

    print(f"Generating dataset with {n_samples} samples...", end="")
    X, y = make_classification(
        n_samples=n_samples, n_features=12, n_informative=10, n_classes=5, random_state=42
    )
    print("done.")

    X_train = X[:max_train_size]
    y_train = y[:max_train_size]
    x_explain = X[max_train_size]

    print(
        f"Will run the explainer on test sets of sizes: {', '.join(map(str, config['train_sizes']))}"
    )
    timings = explain_timed_many_sizes(
        X_train,
        y_train,
        x_explain,
        train_sizes=config["train_sizes"],
        fit_model=config["fit_model"],
        get_explainer=config["get_explainer"],
        verbose=True,
    )

    if save_as_csv:
        filename = f"{config['name']}_{time.time()}.csv"
        print(f"Saving timings to '{filename}'")
        save_timings_as_csv(timings, filename)


def main() -> None:
    """The main entry point of the script."""
    stress_tests: list[StressTestConfig] = [
        {
            "name": "normal_knn",
            "train_sizes": list(range(100_000, 1_000_001, 100_000)),
            "fit_model": get_normal_knn_factory(k=7),
            "get_explainer": KNNExplainer,
        }
    ]

    for stress_test in stress_tests:
        stress_test_model(stress_test, save_as_csv=True)


if __name__ == "__main__":
    main()
