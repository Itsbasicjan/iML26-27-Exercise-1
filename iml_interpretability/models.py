"""The 11 tree models of the experiment, each with its own preprocessing."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

TREE_DEPTHS = [1, 3, 5, 10, None]
ENSEMBLE_DEPTH = 3
ENSEMBLE_SIZES = [3, 10, 100]


def _pipeline(model: ClassifierMixin) -> Pipeline:
    # medians and categories are learned on the training split only
    numeric = SimpleImputer(strategy="median", keep_empty_features=True)
    categorical = Pipeline([
        ("fill_missing", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include="number")),
        ("categorical", categorical, make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([("preprocessing", preprocessing), ("tree", model)])


def get_models(seed: int = 0) -> dict[str, Pipeline]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, Pipeline]
    """
    models = {}
    for depth in TREE_DEPTHS:
        label = str(depth) if depth is not None else "unlimited"
        models[f"decision_tree_depth_{label}"] = _pipeline(DecisionTreeClassifier(max_depth=depth, random_state=seed))
    for n_trees in ENSEMBLE_SIZES:
        models[f"random_forest_{n_trees}_trees"] = _pipeline(
            RandomForestClassifier(n_estimators=n_trees, max_depth=ENSEMBLE_DEPTH, random_state=seed, n_jobs=-1)
        )
        models[f"gradient_boosting_{n_trees}_trees"] = _pipeline(
            GradientBoostingClassifier(n_estimators=n_trees, max_depth=ENSEMBLE_DEPTH, random_state=seed)
        )
    return models
