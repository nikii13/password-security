"""
STEP 6 - System B: machine-learning password classifier
=======================================================
Trains two supervised models on the password features and compares them:

    Logistic Regression  linear model, one weight per feature (interpretable)
    Random Forest        many decision trees, captures feature combinations

What the script does, in order:
    1. Loads password_features.csv and separates FEATURES (X) from the
       LABEL (y). Metadata such as `source` is never shown to the model,
       otherwise it could read the answer instead of learning it.
    2. Cross-validation on the TRAIN split using the 5 `fold` groups made by
       clean_dataset.py: train on 4 folds, evaluate on the 5th, 5 times.
    3. Selects the model with the best mean F1 in cross-validation.
       The test set is NOT used for this choice.
    4. Retrains both models on the whole train split and evaluates them ONCE
       on the test split and on the challenge set (never used for training).
    5. Compares System B with System A (rule_based_predictions.csv) on the
       same rows, and explains the model with feature importances.

Positive class = 1 (predictable / weak).

Input:   data/processed/password_features.csv
         data/processed/rule_based_predictions.csv  (optional, for comparison)
Outputs: models/<model>.joblib                    (trained models, not committed)
         data/processed/ml_predictions.csv        (no raw passwords)
         reports/tables/ml_cv_results.csv
         reports/tables/ml_metrics.csv
         reports/tables/ml_by_stratum.csv
         reports/tables/ml_feature_importance.csv
         reports/tables/system_a_vs_b.csv
         reports/tables/ml_ablation.csv           (only with --ablation)

Run: python -m src.models.train_ml               (about 1-2 minutes)
     python -m src.models.train_ml --ablation    (+ robustness check)
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.common import FEATURES_DATASET, PROCESSED_DIR, REPORTS_DIR, ROOT, SEED

MODELS_DIR = ROOT / "models"
TABLES_DIR = REPORTS_DIR / "tables"
RULE_BASED_PREDICTIONS = PROCESSED_DIR / "rule_based_predictions.csv"
ML_PREDICTIONS = PROCESSED_DIR / "ml_predictions.csv"

# Columns that are NOT features. The model must never see them:
# `source`/`stratum`/`vocab` reveal where the password came from (= the label),
# `rule_compliant` is System A's rule, the rest is bookkeeping.
NON_FEATURE_COLUMNS = ["source", "stratum", "vocab", "label", "label_name",
                       "rule_compliant", "family_id", "split", "fold"]
TARGET = "label"
THRESHOLD = 0.5          # probability >= 0.5 -> predictable


def build_models():
    return {
        # Features have very different scales (length vs 0/1 flags), so the
        # linear model gets a StandardScaler in front of it.
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        # min_samples_leaf keeps trees from memorising single passwords and
        # keeps the saved model small.
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    }


def feature_columns(df):
    """Every column except the metadata ones."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def cross_validation(models, X, y, folds):
    """5-fold CV with the family-aware folds made by clean_dataset.py."""
    cv = PredefinedSplit(folds)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, n_jobs=1,
                                scoring=["accuracy", "precision", "recall", "f1"])
        row = {"model": name}
        for m in ["accuracy", "precision", "recall", "f1"]:
            row[f"{m}_mean"] = scores[f"test_{m}"].mean()
            row[f"{m}_std"] = scores[f"test_{m}"].std()
        rows.append(row)
        print(f"  {name:20s} F1 = {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
    return pd.DataFrame(rows)


def feature_importance(models, X_test, y_test, features):
    """LR: standardised coefficients (sign = direction).
    RF: permutation importance on the test set (drop in F1 when a feature
    is shuffled), which is less biased than the built-in importance."""
    lr = models["logistic_regression"][-1]
    rf = models["random_forest"]
    sample = X_test.sample(min(5000, len(X_test)), random_state=SEED)
    perm = permutation_importance(rf, sample, y_test.loc[sample.index],
                                  scoring="f1", n_repeats=5,
                                  random_state=SEED, n_jobs=-1)
    return (pd.DataFrame({
        "feature": features,
        "lr_coefficient": lr.coef_[0],
        "rf_permutation_importance": perm.importances_mean,
    }).sort_values("rf_permutation_importance", ascending=False))


# Robustness check (optional, --ablation): retrain the Random Forest without
# groups of features and see whether challenge detection survives. If it
# collapsed after removing one feature, the model would rely on a shortcut.
ABLATIONS = {
    "all_features": [],
    "no_class_transitions": ["class_transitions"],
    "no_structure_features": ["class_transitions", "letters_then_digits",
                              "word_digits_symbol", "trailing_digits",
                              "starts_with_upper", "ends_with_symbol"],
    "no_dictionary_features": ["contains_common_word", "common_word_ratio",
                               "leet_used"],
}


def ablation(df, features):
    train, test = df[df.split == "train"], df[df.split == "test"]
    ch = df[df.split == "challenge"]
    rce = test[test.stratum == "rule_compliant_extra"]
    rows = []
    for name, dropped in ABLATIONS.items():
        cols = [c for c in features if c not in dropped]
        model = build_models()["random_forest"].fit(train[cols], train[TARGET])
        rows.append({
            "experiment": name,
            "features_removed": ", ".join(dropped),
            "test_f1": f1_score(test[TARGET], model.predict(test[cols])),
            "challenge_recall_seen": model.predict(ch.loc[ch.vocab == "seen", cols]).mean(),
            "challenge_recall_unseen": model.predict(ch.loc[ch.vocab == "unseen", cols]).mean(),
            "rule_compliant_rockyou_recall": model.predict(rce[cols]).mean(),
        })
        print(f"  {name:24s} done")
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", action="store_true",
                    help="also run the robustness check (about 1 extra minute)")
    args = ap.parse_args()

    df = pd.read_csv(FEATURES_DATASET, keep_default_na=False,
                     dtype={"family_id": str, "vocab": str})
    features = feature_columns(df)
    train = df[df.split == "train"]
    print(f"Train rows: {len(train):,} | features: {len(features)}")

    X_train, y_train = train[features], train[TARGET]

    # 1-3. Cross-validation and model selection (train split only)
    print("\nCross-validation on train (5 family-aware folds):")
    cv_table = cross_validation(build_models(), X_train, y_train, train["fold"])
    best_name = cv_table.sort_values("f1_mean", ascending=False).iloc[0]["model"]
    print(f"Selected by CV: {best_name}")

    # 4. Retrain on the full train split, evaluate once on test and challenge
    models = build_models()
    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    eval_rows, pred_frames = [], []
    for name, model in models.items():
        for subset in ["test", "challenge"]:
            part = df[df.split == subset]
            proba = model.predict_proba(part[features])[:, 1]
            pred = (proba >= THRESHOLD).astype(int)
            eval_rows.append({"model": name, "subset": subset,
                              "selected": name == best_name,
                              **metrics(part[TARGET], pred)})
            pred_frames.append(pd.DataFrame({
                "row_id": part.index, "model": name, "probability": proba,
                "prediction": pred}))
    metrics_table = pd.DataFrame(eval_rows)

    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(ML_PREDICTIONS, index=False)

    # Results per group, for the selected model
    best = preds[preds.model == best_name].set_index("row_id")
    groups = df.loc[best.index, ["split", "source", "stratum", "vocab", TARGET]].join(best)
    by_stratum = (groups.groupby(["split", "source", "stratum", "vocab"])
                  .agg(n=(TARGET, "size"),
                       pct_predicted_predictable=("prediction", "mean"),
                       mean_probability=("probability", "mean"))
                  .reset_index())

    # 5a. System A vs System B on exactly the same rows
    comparison = None
    if RULE_BASED_PREDICTIONS.exists():
        a = pd.read_csv(RULE_BASED_PREDICTIONS, keep_default_na=False).set_index("row_id")
        both = groups.join(a[["strength", "prediction"]].rename(
            columns={"prediction": "system_a_pred", "strength": "system_a_strength"}))
        comparison = (both.groupby(["split", "source", "stratum", "vocab"])
                      .agg(n=(TARGET, "size"),
                           true_label=(TARGET, "first"),
                           system_a_pct_predictable=("system_a_pred", "mean"),
                           system_b_pct_predictable=("prediction", "mean"),
                           a_strong_but_b_predictable=(
                               "prediction",
                               lambda s: ((s == 1) & (both.loc[s.index, "system_a_strength"] == "Strong")).mean()))
                      .reset_index())

    # 5b. Feature importances
    test = df[df.split == "test"]
    importance = feature_importance(models, test[features], test[TARGET], features)

    # Save tables
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(TABLES_DIR / "ml_cv_results.csv", index=False)
    metrics_table.to_csv(TABLES_DIR / "ml_metrics.csv", index=False)
    by_stratum.to_csv(TABLES_DIR / "ml_by_stratum.csv", index=False)
    importance.to_csv(TABLES_DIR / "ml_feature_importance.csv", index=False)
    if comparison is not None:
        comparison.to_csv(TABLES_DIR / "system_a_vs_b.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nTest and challenge metrics (positive class = predictable):")
    print(metrics_table.round(3).to_string(index=False))
    print(f"\nSelected model ({best_name}) by group:")
    print(by_stratum.round(3).to_string(index=False))
    if comparison is not None:
        print("\nSystem A vs System B (% classified as predictable):")
        print(comparison.round(3).to_string(index=False))
    print("\nTop 10 features (Random Forest permutation importance):")
    print(importance.head(10).round(4).to_string(index=False))

    if args.ablation:
        print("\nRobustness check (Random Forest without groups of features):")
        table = ablation(df, features)
        table.to_csv(TABLES_DIR / "ml_ablation.csv", index=False)
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
