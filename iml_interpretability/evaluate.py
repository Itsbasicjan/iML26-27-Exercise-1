"""Evaluate interpretability metrics for every (dataset, model) combination."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from iml_interpretability.data import Dataset
from iml_interpretability.metric import InterpretabilityMetric


def evaluate(
    datasets: Iterable[Dataset],
    models: dict[str, ClassifierMixin],
    metrics: Iterable[InterpretabilityMetric],
    seed: int = 0,
    test_size: float = 0.25,
    positive_labels: dict[str, int | str] | None = None,
) -> pd.DataFrame:
    """Fit every model on every dataset and compute test accuracy and all metrics.

    :param datasets: Datasets to evaluate on.
    :param models: Unfitted models keyed by name, e.g. from :func:`iml_interpretability.models.get_models`.
    :param metrics: Interpretability metrics to compute for each fitted model.
    :param seed: Random seed for the train/test split.
    :param test_size: Fraction of each dataset held out for computing accuracy.
    :param positive_labels: Positive class for binary precision/recall per dataset (default: 1).
    :return: One row per (dataset, model), including test accuracy, precision, recall and interpretability metrics.
    :rtype: pd.DataFrame
    """
    metrics = list(metrics)
    rows = []
    for dataset in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=seed, stratify=dataset.y
        )
        train = Dataset(name=dataset.name, X=X_train, y=y_train)
        positive_label = (positive_labels or {}).get(dataset.name, 1)
        for model_name, model in models.items():
            # clone so every dataset gets a fresh model
            fitted = clone(model).fit(X_train, y_train)
            predictions = fitted.predict(X_test)
            row = {
                "dataset": dataset.name,
                "model": model_name,
                "positive_label": positive_label,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, pos_label=positive_label, zero_division=0),
                "recall": recall_score(y_test, predictions, pos_label=positive_label, zero_division=0),
            }
            for metric in metrics:
                row.update(metric.measure(fitted, train))
            rows.append(row)
    return pd.DataFrame(rows)
