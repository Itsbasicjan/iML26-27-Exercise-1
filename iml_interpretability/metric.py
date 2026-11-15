"""Interpretability metrics.

:class:`CTIS` is the metric used in the experiment. :class:`RelativeNumberOfLeaves` is the
example from the course template and is not used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import exp, log2

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    def measure(self, model, dataset) -> dict[str, float]:
        """Return the score and, optionally, intermediate values for the results table."""
        return {self.name: self.score(model, dataset)}

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


class RelativeNumberOfLeaves(InterpretabilityMetric):
    """Number of leaves of a decision tree relative to the maximum possible for its depth.

    A binary tree of depth ``d`` has at most ``2 ** d`` leaves, so the score lies in ``(0, 1]``, where
    1 means a complete tree. Lower means the tree uses fewer of the decision rules its depth would
    allow. This is only an example: replace it with your own metric, which should go beyond such
    simple size measures, and argue why it captures interpretability.
    """

    name = "relative_n_leaves"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError(f"{type(self).__name__} only supports DecisionTreeClassifier, got {type(model).__name__}")
        return model.get_n_leaves() / 2 ** model.get_depth()


class CTIS(InterpretabilityMetric):
    """CTIS of a fitted tree, random forest or gradient boosting model. Higher = easier to read.

    Path costs are averaged over the observations in ``dataset``. For ensembles they are summed
    over all trees and L counts the leaves of all trees.
    """

    name = "ctis"

    def score(self, model, dataset) -> float:
        return self.measure(model, dataset)[self.name]

    def measure(self, model, dataset) -> dict[str, float]:
        X = dataset.X
        if isinstance(model, Pipeline):
            X = model[:-1].transform(X)
            model = model.steps[-1][1]
        if len(dataset.y) == 0:
            raise ValueError("CTIS requires at least one observation.")

        trees = _fitted_trees(model)
        # leaf of every observation in every tree, shape (n, n_trees); GB returns floats
        leaves_per_tree = np.asarray(model.apply(X)).astype(int).reshape(len(dataset.y), len(trees))

        totals = np.zeros((len(dataset.y), 3))
        for column, tree in enumerate(trees):
            totals += _leaf_costs(tree.tree_)[leaves_per_tree[:, column]]

        depth, switches, repetitions = totals.mean(axis=0)
        mean_p = depth + 0.5 * switches + 0.25 * repetitions
        leaves = int(sum(tree.get_n_leaves() for tree in trees))
        ctis = 100 * exp(-(0.7 * mean_p + 0.3 * log2(1 + leaves)) / 6)
        return {
            "n_trees": len(trees),
            "mean_decisions": float(depth),
            "mean_feature_switches": float(switches),
            "mean_repeated_features": float(repetitions),
            "mean_path_cost": float(mean_p),
            "n_leaves": leaves,
            self.name: ctis,
        }


def _fitted_trees(model) -> list:
    """Fitted trees of the model, in the same order as the columns of model.apply()."""
    if isinstance(model, BaseDecisionTree):
        return [model]
    # GB stores trees as (n_stages, k), ravel() matches the reshape of apply()
    estimators = getattr(model, "estimators_", None)
    if estimators is not None and hasattr(model, "apply"):
        trees = list(np.ravel(estimators))
        if trees and all(isinstance(tree, BaseDecisionTree) for tree in trees):
            return trees
    raise TypeError(
        f"CTIS supports decision trees, random forests and gradient boosting, got {type(model).__name__}."
    )


def _leaf_costs(tree) -> np.ndarray:
    """d, s and r for the path to every leaf."""
    costs = np.zeros((tree.node_count, 3))
    stack = [(0, [])]
    while stack:
        node, features = stack.pop()
        if tree.children_left[node] == tree.children_right[node]:
            depth = len(features)
            switches = sum(a != b for a, b in zip(features, features[1:]))
            repetitions = depth - len(set(features))
            costs[node] = [depth, switches, repetitions]
        else:
            path = features + [int(tree.feature[node])]
            stack.append((tree.children_left[node], path))
            stack.append((tree.children_right[node], path))
    return costs
