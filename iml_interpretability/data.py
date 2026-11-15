"""Dataset loading. The credit data is returned raw, the model pipelines do the preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

# OpenML downloads are cached here, relative to the current working directory.
DATA_HOME = Path("data")


@dataclass(frozen=True)
class Dataset:
    """A tabular classification dataset (credit data are preprocessed by the model pipeline).

    :param name: Human-readable dataset name, used in result tables.
    :param X: Feature matrix; may contain categories and missing values.
    :param y: Target vector.
    """

    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


# Columns that leak the target (boat = lifeboat, body = body identification number) or are
# free-text identifiers without a meaningful encoding.
_TITANIC_DROP_COLUMNS = ["boat", "body", "home.dest", "name", "ticket", "cabin"]


def load_titanic(data_home: str | Path = DATA_HOME) -> Dataset:
    """Load the Titanic dataset from OpenML (data id 40945).

    The download is cached in ``data_home``, so network access is only needed on the first call.

    :param data_home: Cache directory for the download, relative paths are resolved against the
        current working directory.
    :return: The preprocessed Titanic dataset.
    :rtype: Dataset
    """
    # The default pandas ARFF parser crashes on Titanic's integer-valued categorical columns.
    bunch = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff", data_home=data_home)
    X = bunch.data.drop(columns=_TITANIC_DROP_COLUMNS)
    y = bunch.target.astype(int).rename("survived")

    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    return Dataset(name="titanic", X=X, y=y)


def load_credit_datasets(data_home: str | Path = DATA_HOME) -> list[Dataset]:
    """Read the three credit CSVs from data_home."""
    data_home = Path(data_home)
    light = pd.read_csv(data_home / "light_cardclient.csv", sep=";", encoding="utf-8-sig")
    # The first line contains X1, X2, ...; the second contains actual column names.
    mid = pd.read_csv(data_home / "mid_cardclients.csv", header=1, encoding="utf-8-sig")
    hard = pd.read_csv(data_home / "hard_cardclient.csv")

    datasets = []
    for name, frame, target, identifiers in [
        ("light_cardclient", light, "Kreditwuerdigkeit", []),
        ("mid_cardclients", mid, "default payment next month", ["ID"]),
        ("hard_cardclient", hard, "TARGET", ["SK_ID_CURR"]),
    ]:
        if frame[target].isna().any():
            raise ValueError(f"Missing target values in {name}.")
        X = frame.drop(columns=[target, *identifiers])
        datasets.append(Dataset(name=name, X=X, y=frame[target]))
    return datasets
