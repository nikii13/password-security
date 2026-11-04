"""
STEP 5 - System A: traditional rule-based password checker
==========================================================
Reproduces the kind of checker most websites use: it only counts length
and character classes (the "LUDS" model: Lowercase, Uppercase, Digits,
Symbols). It does NOT look at words, years or patterns. That is the point:
it is the baseline the ML model (System B) is compared against.

Rules checked:
    min_length_8     length >= 8
    long_length_12   length >= 12
    has_lowercase    at least one a-z
    has_uppercase    at least one A-Z
    has_digit        at least one 0-9
    has_symbol       at least one non-alphanumeric character

Classification (a typical meter policy):
    Weak    length < 8, or at most 2 character classes
    Medium  length >= 8 and 3 character classes
    Strong  length >= 8 and 4 character classes,
            or length >= 12 and 3 character classes

To compare with the ML model (binary labels), the classes are mapped:
    Weak / Medium -> 1 (predictable)      Strong -> 0 (not predictable)

Input:   data/processed/passwords_clean.csv
Outputs: data/processed/rule_based_predictions.csv   (no raw passwords)
         reports/tables/rule_based_metrics.csv
         reports/tables/rule_based_by_stratum.csv

Run: python -m src.models.rule_based
"""
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.utils.common import CLEAN_DATASET, PROCESSED_DIR, REPORTS_DIR

PREDICTIONS_PATH = PROCESSED_DIR / "rule_based_predictions.csv"
TABLES_DIR = REPORTS_DIR / "tables"

CLASS_TO_BINARY = {"Weak": 1, "Medium": 1, "Strong": 0}


# --------------------------------------------------
# SYSTEM A
# --------------------------------------------------

def check_rules(password):
    """Return {rule name: True/False} for one password."""
    return {
        "min_length_8": len(password) >= 8,
        "long_length_12": len(password) >= 12,
        "has_lowercase": any(c.islower() for c in password),
        "has_uppercase": any(c.isupper() for c in password),
        "has_digit": any(c.isdigit() for c in password),
        "has_symbol": any(not c.isalnum() for c in password),
    }


def classify(password):
    """
    Classify one password as Weak / Medium / Strong.

    Returns a dict with the class, the binary prediction and the lists of
    passed / failed rules (what the web interface will show to the user).
    """
    password = "" if password is None else str(password)
    rules = check_rules(password)
    classes = sum(rules[r] for r in
                  ("has_lowercase", "has_uppercase", "has_digit", "has_symbol"))

    if not rules["min_length_8"] or classes <= 2:
        strength = "Weak"
    elif classes == 4 or (classes == 3 and rules["long_length_12"]):
        strength = "Strong"
    else:
        strength = "Medium"

    return {
        "strength": strength,
        "prediction": CLASS_TO_BINARY[strength],
        "char_classes": classes,
        "passed_rules": [r for r, ok in rules.items() if ok],
        "failed_rules": [r for r, ok in rules.items() if not ok],
        **{f"rule_{r}": int(ok) for r, ok in rules.items()},
    }


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def metrics(y_true, y_pred):
    """Positive class = 1 (predictable / weak)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def main():
    df = pd.read_csv(CLEAN_DATASET, keep_default_na=False,
                     dtype={"password": str, "family_id": str})
    print(f"Classifying {len(df):,} passwords with System A...")

    results = pd.DataFrame(df["password"].map(classify).tolist(), index=df.index)
    keep = ["source", "stratum", "vocab", "label", "split", "fold"]
    out = pd.concat([df[keep], results.drop(columns=["passed_rules", "failed_rules"])],
                    axis=1)
    out.insert(0, "row_id", df.index)        # same row order as password_features.csv
    out.to_csv(PREDICTIONS_PATH, index=False)

    # Metrics on the test split and on the challenge set.
    # System A has no training, so the train split is not needed here.
    rows = []
    for name, part in [("test", out[out.split == "test"]),
                       ("challenge", out[out.split == "challenge"])]:
        rows.append({"subset": name, **metrics(part.label, part.prediction)})
    table = pd.DataFrame(rows)

    by_stratum = (out[out.split.isin(["test", "challenge"])]
                  .groupby(["split", "source", "stratum", "vocab"])
                  .agg(n=("label", "size"),
                       pct_weak=("strength", lambda s: (s == "Weak").mean()),
                       pct_medium=("strength", lambda s: (s == "Medium").mean()),
                       pct_strong=("strength", lambda s: (s == "Strong").mean()))
                  .reset_index())

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "rule_based_metrics.csv", index=False)
    by_stratum.to_csv(TABLES_DIR / "rule_based_by_stratum.csv", index=False)

    pd.set_option("display.width", 140)
    print("\nSystem A metrics (positive class = predictable):")
    print(table.round(3).to_string(index=False))
    print("\nStrength distribution by subset and stratum:")
    print(by_stratum.round(3).to_string(index=False))
    print(f"\nPredictions saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
