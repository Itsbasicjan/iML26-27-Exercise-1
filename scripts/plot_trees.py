"""Plot the depth-3 tree of every dataset.

Usage: uv run python -m scripts.plot_trees
"""

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from scripts.tree_labels import TARGETS, question

from iml_interpretability import CTIS, Dataset, get_models, load_credit_datasets


def draw_tree(ax, tree, features, dataset):
    """Draw every split as a yes/no question."""
    structure = tree.tree_
    depth_limit = tree.get_depth()
    categories = dataset.X.select_dtypes(exclude="number").columns

    def draw(node, depth, left, right):
        x = (left + right) / 2
        y = 1 - depth / (depth_limit + 1)
        count = structure.n_node_samples[node]
        is_leaf = structure.children_left[node] == structure.children_right[node]
        if is_leaf:
            values = structure.value[node][0]
            index = values.argmax()
            prediction = TARGETS[dataset.name][tree.classes_[index]]
            share = values[index] / values.sum()
            label = "Modellvorhersage:\n" + fill(prediction, 30)
            label += f"\nTrainingsfälle: {count:,}".replace(",", ".")
            label += f"\nAnteil dieser Klasse: {share:.1%}".replace(".", ",")
            color = "#e1eff1"
        else:
            feature = features[structure.feature[node]]
            text, yes_left = question(feature, structure.threshold[node], categories)
            label = fill(text, 32) + f"\nTrainingsfälle: {count:,}".replace(",", ".")
            color = "#e8edf7"
            for child, lo, hi, branch in [
                (structure.children_left[node], left, x, "Ja" if yes_left else "Nein"),
                (structure.children_right[node], x, right, "Nein" if yes_left else "Ja"),
            ]:
                child_x, child_y = (lo + hi) / 2, 1 - (depth + 1) / (depth_limit + 1)
                ax.annotate("", (child_x, child_y), (x, y),
                            arrowprops={"arrowstyle": "->", "color": "#687386"}, zorder=1)
                ax.text(x + 0.6 * (child_x - x), y + 0.6 * (child_y - y), branch,
                        ha="center", va="center", fontsize=11,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=3)
                draw(child, depth + 1, lo, hi)
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "#a0aabc"}, zorder=4)

    draw(0, 0, 0, 1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.15)
    ax.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export three complete depth-3 trees as PNG.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/trees"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in load_credit_datasets(args.data_dir):
        # same split as run_experiment.py
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=0.25, random_state=0, stratify=dataset.y
        )
        model = get_models(seed=0)["decision_tree_depth_3"]
        model.fit(X_train, y_train)
        tree = model.named_steps["tree"]
        features = model.named_steps["preprocessing"].get_feature_names_out()
        score = CTIS().score(model, Dataset(dataset.name, X_train, y_train))
        accuracy = accuracy_score(y_test, model.predict(X_test))

        depth = tree.get_depth()
        fig, ax = plt.subplots(figsize=(max(11, 4 * 2 ** depth), 4 + 2 * depth))
        draw_tree(ax, tree, features, dataset)
        title = f"{dataset.name} | Maximale Baumtiefe: 3 | CTIS: {score:.2f} | Test-Accuracy: {accuracy:.1%}"
        fig.suptitle(title + "\nJa / Nein beantwortet die Frage am jeweiligen Knoten", fontsize=13)
        fig.text(0.5, 0.025, "Modellvorhersage ≠ Kreditentscheidung · Klassenanteile aus Trainingsdaten · Schwellenwerte gerundet",
                 ha="center", fontsize=11)
        fig.tight_layout(rect=(0, 0.07, 1, 0.9))
        output = args.output_dir / f"{dataset.name}_depth_3.png"
        fig.savefig(output, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output} (CTIS={score:.6f}, accuracy={accuracy:.6f})", flush=True)


if __name__ == "__main__":
    main()
