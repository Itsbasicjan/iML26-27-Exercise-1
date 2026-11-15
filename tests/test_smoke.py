"""Smoke tests that run offline on a synthetic dataset."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import Dataset, InterpretabilityMetric, RelativeNumberOfLeaves, evaluate, get_models


class _ConstantMetric(InterpretabilityMetric):
    name = "constant"

    def score(self, model, dataset):
        return 1.0


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_evaluate_returns_one_row_per_dataset_and_model(dataset):
    models = get_models()
    results = evaluate([dataset], models, [_ConstantMetric()])
    assert len(results) == len(models)
    assert set(results.columns) == {"dataset", "model", "positive_label", "accuracy", "precision", "recall", "constant"}


@pytest.mark.parametrize("prediction, accuracy, precision, recall", [(2, 0.4, 0.4, 1), (1, 0.6, 0, 0)])
def test_risk_class_metrics(prediction, accuracy, precision, recall):
    dataset = Dataset("credit", pd.DataFrame({"x": range(20)}), pd.Series([1] * 12 + [2] * 8))
    result = evaluate(
        [dataset], {"constant": DummyClassifier(strategy="constant", constant=prediction)}, [],
        positive_labels={"credit": 2},
    ).iloc[0]
    assert result["positive_label"] == 2
    assert result["accuracy"] == pytest.approx(accuracy)
    assert result["precision"] == pytest.approx(precision)
    assert result["recall"] == pytest.approx(recall)


def test_relative_number_of_leaves_matches_tree(dataset):
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(dataset.X, dataset.y)
    score = RelativeNumberOfLeaves().score(model, dataset)
    assert score == model.get_n_leaves() / 2 ** model.get_depth()
    assert 0 < score <= 1


def test_relative_number_of_leaves_rejects_non_tree_models(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        RelativeNumberOfLeaves().score(model, dataset)
