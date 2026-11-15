"""Small trees with path costs that can be checked by hand."""

from math import exp, log2

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import CTIS, Dataset


def test_repeated_feature_and_observation_weighting():
    # one observation needs 1 decision, three need 2 decisions on the same feature
    dataset = Dataset("repeat", pd.DataFrame({"x": [0, 1, 2, 3]}), pd.Series([0, 1, 1, 0]))
    model = DecisionTreeClassifier(random_state=0).fit(dataset.X, dataset.y)
    result = CTIS().measure(model, dataset)
    assert result["mean_decisions"] == 1.75
    assert result["mean_feature_switches"] == 0
    assert result["mean_repeated_features"] == 0.75
    assert result["n_leaves"] == 3
    expected_p = 1.75 + 0.25 * 0.75
    assert result["ctis"] == pytest.approx(100 * exp(-(0.7 * expected_p + 0.3 * 2) / 6))


def test_switch_between_features():
    dataset = Dataset(
        "xor", pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}), pd.Series([0, 1, 1, 0])
    )
    model = DecisionTreeClassifier(random_state=0).fit(dataset.X, dataset.y)
    result = CTIS().measure(model, dataset)
    assert result["mean_decisions"] == 2
    assert result["mean_feature_switches"] == 1
    assert result["mean_repeated_features"] == 0
    assert result["ctis"] == pytest.approx(100 * exp(-(0.7 * 2.5 + 0.3 * log2(5)) / 6))


def test_tree_without_decisions():
    dataset = Dataset("constant", pd.DataFrame({"x": [0, 1]}), pd.Series([1, 1]))
    model = DecisionTreeClassifier().fit(dataset.X, dataset.y)
    result = CTIS().measure(model, dataset)
    assert result["mean_path_cost"] == 0
    assert result["n_leaves"] == 1
    assert result["ctis"] == pytest.approx(100 * exp(-0.3 / 6))


def test_gradient_boosting_sums_path_costs_over_trees():
    # two stumps: one decision per tree, no switch between trees
    dataset = Dataset("stumps", pd.DataFrame({"x": [0, 1, 2, 3]}), pd.Series([0, 0, 1, 1]))
    model = GradientBoostingClassifier(n_estimators=2, max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    result = CTIS().measure(model, dataset)
    assert result["n_trees"] == 2
    assert result["mean_decisions"] == 2
    assert result["mean_feature_switches"] == 0
    assert result["n_leaves"] == 4
    assert result["ctis"] == pytest.approx(100 * exp(-(0.7 * 2 + 0.3 * log2(5)) / 6))


def test_forest_costs_equal_sum_of_its_trees():
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    dataset = Dataset("synthetic", pd.DataFrame(X), pd.Series(y))
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(dataset.X, dataset.y)
    result = CTIS().measure(forest, dataset)
    per_tree = [CTIS().measure(tree, Dataset("t", pd.DataFrame(X.astype("float32")), dataset.y))
                for tree in forest.estimators_]
    for key in ["mean_decisions", "mean_feature_switches", "mean_repeated_features", "n_leaves"]:
        assert result[key] == pytest.approx(sum(r[key] for r in per_tree))
    assert result["ctis"] < min(r["ctis"] for r in per_tree)


def test_unsupported_model_is_rejected():
    dataset = Dataset("constant", pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]))
    with pytest.raises(TypeError):
        CTIS().measure(LogisticRegression().fit(dataset.X, dataset.y), dataset)
