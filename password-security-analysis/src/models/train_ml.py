"""ML training template. Set TARGET_COLUMN to the real existing label."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = Path("data/processed/password_features.csv")
TARGET_COLUMN = "label"  # Change to the actual label column.

def main():
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"No existe la etiqueta {TARGET_COLUMN!r}. "
            "Acordad primero qué columna es el target."
        )

    # Exclude identifiers and any raw password/text columns from X.
    excluded = [TARGET_COLUMN, "password"]
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(random_state=42, class_weight="balanced")
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()
