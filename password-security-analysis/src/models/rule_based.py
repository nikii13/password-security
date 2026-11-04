"""Illustrative baseline only; agree and justify thresholds with the team."""
from src.features.password_features import extract_password_features

def classify_password(password: str) -> str:
    f = extract_password_features(password)

    # Placeholder policy. These thresholds are not a universal security standard.
    if f["length"] >= 14 and sum(
        f[k] for k in ("has_lower", "has_upper", "has_digit", "has_symbol")
    ) >= 3:
        return "strong"
    if f["length"] >= 10:
        return "medium"
    return "weak"

if __name__ == "__main__":
    examples = ["summer2024", "A-longer-example!92", "abc"]
    for example in examples:
        # Do not use real user passwords in console logs.
        print(classify_password(example))
