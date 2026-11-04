"""Shared feature extraction for Rule-Based and ML.

Never log or export the raw password column unnecessarily. The estimated
character-pool entropy below is a rough heuristic, not true password entropy.
"""
from pathlib import Path
import math
import string
import pandas as pd

def extract_password_features(password: str) -> dict:
    password = "" if password is None else str(password)
    length = len(password)
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(not c.isalnum() for c in password)
    pool = (
        (26 if has_lower else 0)
        + (26 if has_upper else 0)
        + (10 if has_digit else 0)
        + (32 if has_symbol else 0)
    )
    estimated_entropy = length * math.log2(pool) if pool > 0 else 0.0

    return {
        "length": length,
        "lower_count": sum(c.islower() for c in password),
        "upper_count": sum(c.isupper() for c in password),
        "digit_count": sum(c.isdigit() for c in password),
        "symbol_count": sum(not c.isalnum() for c in password),
        "has_lower": int(has_lower),
        "has_upper": int(has_upper),
        "has_digit": int(has_digit),
        "has_symbol": int(has_symbol),
        "unique_char_ratio": len(set(password)) / length if length else 0.0,
        "estimated_pool_entropy": estimated_entropy,
    }

def build_feature_dataframe(df: pd.DataFrame, password_column: str = "password"):
    if password_column not in df.columns:
        raise ValueError(f"Missing column: {password_column}")
    features = df[password_column].apply(extract_password_features).apply(pd.Series)
    return pd.concat([df.drop(columns=[password_column]), features], axis=1)

def main():
    input_path = Path("data/processed/passwords_clean.csv")
    output_path = Path("data/processed/password_features.csv")
    df = pd.read_csv(input_path)
    result = build_feature_dataframe(df, "password")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Features guardadas en {output_path}")

if __name__ == "__main__":
    main()
