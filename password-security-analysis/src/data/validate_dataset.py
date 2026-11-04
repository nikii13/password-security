"""Basic quality report; does not print password values."""
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/processed/passwords_clean.csv")
PASSWORD_COLUMN = "password"

def main():
    df = pd.read_csv(INPUT_PATH)
    if PASSWORD_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {PASSWORD_COLUMN}")

    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_by_column": df.isna().sum().to_dict(),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "empty_password_rows": int((df[PASSWORD_COLUMN].astype(str).str.len() == 0).sum()),
    }
    print(report)

if __name__ == "__main__":
    main()
