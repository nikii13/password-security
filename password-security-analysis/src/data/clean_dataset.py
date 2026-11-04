"""Initial cleaning template. Review decisions with the dataset owner first."""
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/samples/password_sample.csv")
OUTPUT_PATH = Path("data/processed/passwords_clean.csv")
PASSWORD_COLUMN = "password"  # Change to the actual source column.

def main():
    df = pd.read_csv(INPUT_PATH)

    # Basic structural checks
    if PASSWORD_COLUMN not in df.columns:
        raise ValueError(
            f"Falta la columna {PASSWORD_COLUMN!r}. Columnas: {list(df.columns)}"
        )

    # Remove missing password rows; document this decision in the report.
    df = df.dropna(subset=[PASSWORD_COLUMN]).copy()

    # Convert to string without silently changing case or trimming passwords:
    # spaces and case may be meaningful password characters.
    df[PASSWORD_COLUMN] = df[PASSWORD_COLUMN].astype(str)

    # Remove exact duplicate rows only. Decide separately whether repeated
    # password values should be retained for frequency analysis.
    df = df.drop_duplicates()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset guardado: {OUTPUT_PATH} ({len(df):,} filas)")

if __name__ == "__main__":
    main()
