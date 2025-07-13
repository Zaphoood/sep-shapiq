"""Visualization utilities for interpreting KNN-based Shapley values in 2D."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable


BASE_SCALE = 100


def plot_knn_shapley_2d(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.integer],
    shapley_values: npt.NDArray[np.floating],
    classes: npt.NDArray[np.integer],
    x_test: npt.NDArray[np.floating] | None = None,
    *,
    title: str | None = None,
    scale: float = 1,
) -> None:
    """Plot a 2D KNN Shapley explanation using matplotlib.

    This function visualizes the training data, a validation point, and
    the associated Shapley values computed for a KNN explanation.
    Each training point is shown with a marker size proportional to the
    absolute Shapley value, and each point is annotated with its numeric
    Shapley score. The validation point is highlighted with an empty circle
    in the color corresponding to its predicted class.

    Args:
        X_train: A 2D array of shape (n_samples, 2) with the training points.
        y_train: A 1D array of shape (n_samples,) with class labels for each training point.
        shapley_values: A sequence of Shapley values (or similar interaction scores)
                        corresponding to each training point.
        classes: A sequence of unique class labels (must not exceed the number of matplotlib colors available).
        x_test: A array of shape (2,) representing the test datat point being explained.
        title: The title to set on the plot.
        scale: Scaling factor for Shapley marker sizes (default is 3).

    Raises:
        ValueError: If all Shapley values are zero.
        RuntimeError: If the number of classes exceeds the number of available default colors.
    """
    if x_test is not None:
        x_test = np.atleast_2d(x_test)

    # TODO(Zaphoood): this linter error is so stupid. Like, 2 is just not a magic value!!
    two = 2
    if X_train.ndim != two or X_train.shape[1] != two:
        msg = f"X_train must be 2D matrix with shape (n, 2), but got {X_train.shape}"
        raise ValueError(msg)

    sizes = np.abs(shapley_values)
    if np.max(sizes) == 0:
        msg = "Shapley values must not all be zero."
        raise ValueError(msg)

    sizes = scale * BASE_SCALE * sizes / np.max(sizes)

    plt.figure(figsize=(8, 8))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(classes) > len(colors):
        msg = f"Sorry, too many classes ({len(classes)}) and not enough colors ({len(colors)})!"
        raise RuntimeError(msg)

    for color, class_ in zip(colors, classes, strict=False):
        mask = y_train == class_
        plt.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            sizes[mask],
            label=str(class_),
            facecolors=color,
        )
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=scale * BASE_SCALE,
        marker="o",
        edgecolors="black",
        facecolors="none",
    )

    if x_test is not None:
        plt.scatter(
            x_test[:, 0],
            x_test[:, 1],
            marker="x",
            facecolors="black",
            s=0.5 * scale * BASE_SCALE,
            linewidths=1.5,
        )

    x_lims, y_lims = _axis_lims_center_mean(
        np.vstack([X_train, x_test]) if x_test is not None else X_train
    )
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    plt.gca().set_aspect("equal")
    plt.legend(
        handles=_make_legend_handles(
            colors,
            classes,
            marker_kwargs={
                "marker": "o",
                "color": "w",
            },
            x_test_kwargs={
                "marker": "x",
                "color": "w",
                "label": "x_explain",
                "markeredgecolor": "black",
            },
            legend_size=100,
        )
    )

    if title is not None:
        plt.title(title)


def _make_legend_handles(
    colors: Iterable[str],
    classes: Iterable[int],
    marker_kwargs: dict[str, Any],
    x_test_kwargs: dict[str, Any] | None = None,
    legend_size: float = 100,
) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            markersize=np.sqrt(legend_size),  # markersize is in points (approx sqrt of s)
            label=f"class {class_}",
            markerfacecolor=color,
            **marker_kwargs,
        )
        for color, class_ in zip(colors, classes, strict=False)
    ]
    if x_test_kwargs is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                markersize=np.sqrt(0.5 * legend_size),
                **x_test_kwargs,
            )
        )
    return handles


def _axis_lims_center_origin(
    points: npt.NDArray[np.floating], padding_percent: float = 0.2
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Calculate the x and y limits such that the plot is square and centered around the origin."""
    max_extent = np.abs(points).max()
    limits = max_extent * (1 + padding_percent)

    return (-limits, limits), (-limits, limits)


def _axis_lims_center_mean(
    points: npt.NDArray[np.floating], padding_percent: float = 0.2
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Calcualte the x and y limits such that the plot is square and centered around the mean of the points."""
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    half_range = max(x_max - x_min, y_max - y_min) / 2
    padding_scale = 1 + padding_percent
    x_lims = (x_center - padding_scale * half_range, x_center + padding_scale * half_range)
    y_lims = (y_center - padding_scale * half_range, y_center + padding_scale * half_range)

    return x_lims, y_lims
