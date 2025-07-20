"""Script for plotting coalition finding benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from pandas.core.api import DataFrame


def load_data(filepath) -> DataFrame:
    """Load the CSV data into a pandas DataFrame."""
    return pd.read_csv(filepath)


def validate_constants(df, columns):
    """Check that given columns have only one unique value. Returns a dict of values and warns if inconsistent."""
    constants = {}
    for col in columns:
        unique_vals = df[col].unique()
        constants[col] = unique_vals
        if len(unique_vals) > 1:
            print(f"Warning: '{col}' has multiple values: {unique_vals}")
    return constants


def compute_mean_score(df):
    """Add a column for the mean of avg_min_score and avg_max_score."""
    df["mean_score"] = (df["avg_min_score"] + df["avg_max_score"]) / 2
    return df


def plot_scores(df, constants, *, save=False, output_filename="plot.png"):
    """Plot mean score vs explanation order, grouped by strategy and coal_size."""
    plt.figure(figsize=(10, 7))

    # Use default matplotlib color cycle
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    strategies = df["strategy"].unique()
    strategy_colors = {
        strategy: default_colors[i % len(default_colors)] for i, strategy in enumerate(strategies)
    }

    # Plot each (strategy, coal_size) line without legend
    for (strategy, _), group in df.groupby(["strategy", "coal_size"]):
        group_sorted = group.sort_values("explanation_order")
        plt.plot(
            group_sorted["explanation_order"],
            group_sorted["mean_score"],
            color=strategy_colors[strategy],
            alpha=0.15,
            linewidth=1.5,
            label=None,  # Don't clutter legend
        )

    # Plot average over coal_sizes per strategy (bold)
    avg_df = df.groupby(["strategy", "explanation_order"])["mean_score"].mean().reset_index()
    for strategy, group in avg_df.groupby("strategy"):
        group_sorted = group.sort_values("explanation_order")
        plt.plot(
            group_sorted["explanation_order"],
            group_sorted["mean_score"],
            color=strategy_colors[strategy],
            linewidth=2,
            alpha=1,
            marker="o",
            label=f"{strategy} (avg)",
        )

    plt.xlabel("Explanation Order")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)

    # Set x-axis ticks to integer steps
    min_order = int(df["explanation_order"].min())
    max_order = int(df["explanation_order"].max())
    plt.xticks(range(min_order, max_order + 1))

    # Construct plot title
    title_parts = [
        f"{key}={constants[key][0]}" for key in ["n_players", "n_games"] if len(constants[key]) == 1
    ]

    coal_sizes = df["coal_size"].unique()
    coal_range_str = (
        f"coal_size={coal_sizes.min()},...,{coal_sizes.max()}"
        if len(coal_sizes) > 1
        else f"coal_size={coal_sizes[0]}"
    )
    title_parts.append(coal_range_str)

    title = "Strategy Scores by Explanation Order\n(" + ", ".join(title_parts) + ")"
    plt.title(title)

    # Custom legend
    from matplotlib.lines import Line2D

    strategy_handles_avg = [
        Line2D([0], [0], color=strategy_colors[strategy], lw=2, label=f"{strategy} (avg)")
        for strategy in strategies
    ]
    strategy_handles_coal_sizes = [
        Line2D(
            [0],
            [0],
            color=strategy_colors[strategy],
            lw=1,
            alpha=0.4,
            label=f"{strategy}",
        )
        for strategy in strategies
    ]
    plt.legend(
        handles=strategy_handles_avg + strategy_handles_coal_sizes, title="Legend", fontsize="small"
    )

    plt.grid(visible=True)
    plt.tight_layout()

    if save:
        overwrite = "y"
        if Path(output_filename).is_file():
            overwrite = (
                input(f"File '{output_filename}' already exists. Overwrite? (y/n): ")
                .strip()
                .lower()
            )
        if overwrite == "y":
            plt.savefig(output_filename, dpi=200)
            print(f"Plot saved to '{output_filename}'")
        else:
            print("Aborted saving plot.")
    else:
        plt.show()


def plot_time_vs_players(df, *, save=False, output_filename="time_plot.png"):
    """Plot t_delta vs number of players, grouped by strategy."""
    plt.figure(figsize=(10, 7))

    time_units = "ms"
    time_multiplier = 1000

    # Use default matplotlib colors
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    strategies = df["strategy"].unique()
    strategy_colors = {
        strategy: default_colors[i % len(default_colors)] for i, strategy in enumerate(strategies)
    }

    for strategy, group in df.groupby("strategy"):
        group_sorted = group.sort_values("n_players")
        plt.plot(
            group_sorted["n_players"],
            group_sorted["t_delta"] * time_multiplier,
            marker="o",
            label=strategy,
            color=strategy_colors[strategy],
        )

    plt.xlabel("Number of Players")
    plt.ylabel(f"Time [{time_units}]")
    plt.grid(visible=True)

    # Construct plot title including n_games, explanation_order, coal_size
    title_parts = []
    for key in ["n_games", "explanation_order", "coal_size"]:
        unique_vals = df[key].unique()
        if len(unique_vals) == 1:
            title_parts.append(f"{key}={unique_vals[0]}")
        else:
            # If multiple values, show range
            title_parts.append(f"{key}=[{unique_vals.min()}, {unique_vals.max()}]")

    title = "Time vs Number of Players\n(" + ", ".join(title_parts) + ")"
    plt.title(title)

    plt.legend(title="Strategy")
    plt.tight_layout()

    if save:
        plt.savefig(output_filename)
        print(f"Plot saved to '{output_filename}'")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot strategy benchmark results from a CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV file with benchmark results.")
    parser.add_argument(
        "--save", action="store_true", help="Save the plot to a file instead of displaying it."
    )
    parser.add_argument(
        "--output", default="plot.png", help="Filename for the saved plot (default: plot.png)."
    )
    parser.add_argument(
        "--plot-type",
        choices=["score", "time"],
        default="score",
        help="Type of plot to generate: 'score' (default) or 'time'",
    )

    args = parser.parse_args()

    if not Path(args.csv_path).is_file():
        print(f"Error: File '{args.csv_path}' does not exist.")
        return

    df = load_data(args.csv_path)
    constants = validate_constants(df, ["n_players", "n_games"])
    df = compute_mean_score(df)

    if args.plot_type == "score":
        plot_scores(df, constants, save=args.save, output_filename=args.output)
    elif args.plot_type == "time":
        plot_time_vs_players(df, save=args.save, output_filename=args.output)
    else:
        print(f"Unknown plot type '{args.plot_type}'. Allowed values are 'score' and 'time'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
