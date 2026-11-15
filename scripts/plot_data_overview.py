"""Create two static English dataset charts from the three input CSVs.

Usage: uv run python -m scripts.plot_data_overview
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from iml_interpretability import load_credit_datasets


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(labelsize=12)


def main():
    datasets = load_credit_datasets()
    output = Path("results/data_plots")
    output.mkdir(parents=True, exist_ok=True)
    names = ["Light", "Mid", "Hard"]
    colors = ["#2563eb", "#d97706", "#0f766e"]
    records = [len(d.X) for d in datasets]
    features = [len(d.X.columns) for d in datasets]
    missing = [100 * d.X.isna().sum().sum() / d.X.size for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.16, wspace=0.3)
    fig.suptitle("Dataset overview", x=0.07, ha="left", fontsize=28, y=0.96)
    for ax, title, values, labels in zip(axes, ["Records", "Input features", "Missing values (%)"],
                                        [records, features, missing],
                                        [[f"{v:,}" for v in records], [str(v) for v in features],
                                         [f"{v:.1f}%" for v in missing]]):
        bars = ax.bar(names, values, color=colors, width=0.6)
        ax.bar_label(bars, labels=labels, padding=7, fontsize=12)
        ax.set_ylim(0, max(values) * 1.23)
        ax.set_title(title, fontsize=19, pad=18)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        style(ax)
    fig.savefig(output / "01_dataset_overview.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.16, wspace=0.22)
    fig.suptitle("Target distribution", x=0.07, ha="left", fontsize=28, y=0.96)
    classes = [
        [(1, "Good credit\nrisk"), (2, "Bad credit\nrisk")],
        [(0, "No default"), (1, "Default")],
        [(0, "No payment\ndifficulties"), (1, "Payment\ndifficulties")],
    ]
    for ax, dataset, name, labels in zip(axes, datasets, names, classes):
        counts = dataset.y.value_counts()
        values = [100 * counts[value] / len(dataset.y) for value, _ in labels]
        bars = ax.bar([label for _, label in labels], values, color=["#287f85", "#bd7330"], width=0.6)
        ax.bar_label(bars, labels=[f"{share:.1f}%" for share in values], padding=7, fontsize=12)
        ax.set_title(name, fontsize=19, pad=18)
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        style(ax)
    axes[0].set_ylabel("Percent", fontsize=14)
    fig.savefig(output / "02_target_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
