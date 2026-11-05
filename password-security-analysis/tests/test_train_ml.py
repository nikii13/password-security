"""
Unit tests for System B (src/models/train_ml.py).
Run from the repository root:  python -m pytest -q
"""
import pandas as pd

from src.features.password_features import extract_features
from src.models.train_ml import (NON_FEATURE_COLUMNS, build_models,
                                 feature_columns, metrics)


def tiny_dataset():
    """A handful of passwords with features, in the same format as
    password_features.csv."""
    weak = ["Summer2026!", "maria123", "password1", "Welcome1!", "qwerty12", "iloveyou"]
    strong = ["xK9#mQ2vLp8$", "qhzmtwxkpafr", "k8m2xq7wz4p", "Tz4@Lw9!qR2s",
              "pa0x9vhq2mzr", "R7v#Nq2!Lm8x"]
    rows = []
    for pw, label in [(p, 1) for p in weak] + [(p, 0) for p in strong]:
        rows.append({"source": "rockyou" if label else "random", "stratum": "",
                     "vocab": "", "label": label, "label_name": "", "rule_compliant": 0,
                     "family_id": pw, "split": "train", "fold": 0,
                     **extract_features(pw)})
    return pd.DataFrame(rows)


def test_metadata_is_never_a_feature():
    """Leakage guard: source/label/etc. must not reach the model."""
    features = feature_columns(tiny_dataset())
    assert not set(features) & set(NON_FEATURE_COLUMNS)
    assert "length" in features and "contains_year" in features


def test_models_train_and_predict_probabilities():
    df = tiny_dataset()
    features = feature_columns(df)
    for name, model in build_models().items():
        model.fit(df[features], df["label"])
        proba = model.predict_proba(df[features])[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all(), name


def test_metrics_positive_class_is_predictable():
    m = metrics(pd.Series([1, 1, 0, 0]), pd.Series([1, 0, 0, 0]))
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 0, 2)
    assert m["recall"] == 0.5
    assert m["precision"] == 1.0
