"""Charts 01-04 for the single trees by depth, 05-06 for the ensembles by number of trees."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    results = pd.read_csv("results/scores.csv")
    depths = ["1", "3", "5", "10", "unlimited"]
    x = np.arange(len(depths))
    width = 0.25
    output = Path("results/metric_plots")
    output.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("light_cardclient", "Light", "#2563eb"),
        ("mid_cardclients", "Mid", "#d97706"),
        ("hard_cardclient", "Hard", "#0f766e"),
    ]
    for order, (metric, title) in enumerate([
        ("ctis", "CTIS"), ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ], start=1):
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.16)
        for offset, (dataset, label, color) in enumerate(datasets):
            scores = results[results["dataset"] == dataset].set_index("model")[metric]
            scale = 1 if metric == "ctis" else 100
            values = [scale * scores[f"decision_tree_depth_{depth}"] for depth in depths]
            bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=12, padding=7)

        fig.suptitle(title, x=0.07, ha="left", fontsize=28, y=0.96)
        ax.set_xlabel("Maximum tree depth", labelpad=10, fontsize=14)
        ax.set_ylabel("CTIS score" if metric == "ctis" else "Percent", fontsize=14)
        ax.set_xticks(x, ["1", "3", "5", "10", "Unlimited"])
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=12)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.97, 0.96), ncols=3, fontsize=14)
        fig.savefig(output / f"{order:02d}_{metric}.png", dpi=150)
        plt.close(fig)

    plot_ensembles(results, datasets, output)


def plot_ensembles(results: pd.DataFrame, datasets: list, output: Path) -> None:
    """"1 tree" is the single depth-3 decision tree."""
    sizes = [1, 3, 10, 100]
    x = np.arange(len(sizes))
    width = 0.25
    for order, (metric, title) in enumerate([("ctis", "CTIS of tree ensembles (depth 3)"),
                                             ("accuracy", "Accuracy of tree ensembles (depth 3)")], start=5):
        fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.14, wspace=0.12)
        for ax, (family, family_title) in zip(axes, [("random_forest", "Random forest"),
                                                      ("gradient_boosting", "Gradient boosting")]):
            for offset, (dataset, label, color) in enumerate(datasets):
                scores = results[results["dataset"] == dataset].set_index("model")[metric]
                names = ["decision_tree_depth_3"] + [f"{family}_{n}_trees" for n in sizes[1:]]
                scale = 1 if metric == "ctis" else 100
                values = [scale * scores[name] for name in names]
                bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
                ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=11, padding=5)
            ax.set_title(family_title, fontsize=19, pad=14)
            ax.set_xticks(x, [str(n) for n in sizes])
            ax.set_xlabel("Number of trees", labelpad=10, fontsize=14)
            ax.set_ylim(0, 108)
            ax.set_yticks(range(0, 101, 20))
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=12)
        axes[0].set_ylabel("CTIS score" if metric == "ctis" else "Percent", fontsize=14)
        fig.suptitle(title, x=0.07, ha="left", fontsize=28, y=0.96)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.97, 0.96), ncols=3, fontsize=14)
        fig.savefig(output / f"{order:02d}_ensemble_{metric}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
