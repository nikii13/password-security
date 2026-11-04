"""
STEP 3 - Validate the clean dataset
===================================
Automatic checks that the dataset is correct BEFORE extracting features or
training. If any check fails, the script exits with an error, so nobody
trains a model on a broken dataset.

Input: data/processed/passwords_clean.csv
Run:   python -m src.data.validate_dataset
"""
import math
import sys

import pandas as pd

from src.utils.common import CLEAN_DATASET, is_rule_compliant, is_valid_password
from src.utils.wordlists import CHALLENGE_WORDS_UNSEEN, COMMON_WORDS

REQUIRED = ["password", "source", "label", "family_id", "split", "fold"]
CLASS_SIZES = {"lower": 26, "upper": 26, "digit": 10, "symbol": 24}


def log10_search_space(pw):
    """Brute-force search space from the character classes present."""
    size = (CLASS_SIZES["lower"] * any(c.islower() for c in pw)
            + CLASS_SIZES["upper"] * any(c.isupper() for c in pw)
            + CLASS_SIZES["digit"] * any(c.isdigit() for c in pw)
            + CLASS_SIZES["symbol"] * any(not c.isalnum() for c in pw))
    return len(pw) * math.log10(size)


def run_checks(df):
    """Return a list of (check name, passed, detail)."""
    train, test = df[df.split == "train"], df[df.split == "test"]
    ch = df[df.split == "challenge"]
    leaked = set(train.family_id) & set(test.family_id)
    test_share = len(test) / max(len(train) + len(test), 1)
    balance = df[df.split != "challenge"].label.mean()

    return [
        ("required columns present",
         all(c in df.columns for c in REQUIRED), str(REQUIRED)),
        ("no duplicate passwords",
         not df.password.duplicated().any(),
         f"{df.password.duplicated().sum()} duplicates"),
        ("all passwords valid (length, characters)",
         df.password.map(is_valid_password).all(), ""),
        ("labels are only 0/1",
         set(df.label.unique()) <= {0, 1}, str(sorted(df.label.unique()))),
        ("labels match sources (rockyou/challenge=1, random=0)",
         ((df.source == "random") == (df.label == 0)).all(), ""),
        ("challenge rows only in the challenge split",
         (ch.source == "challenge").all() and len(ch) == (df.source == "challenge").sum(), ""),
        ("every challenge password passes the LUDS rules",
         ch.password.map(is_rule_compliant).all(), ""),
        ("no family in both train and test (no leakage)",
         not leaked, f"{len(leaked)} leaked families"),
        ("test share close to 20%",
         0.15 <= test_share <= 0.25, f"{test_share:.1%}"),
        ("classes roughly balanced (40-60% predictable)",
         0.40 <= balance <= 0.60, f"{balance:.1%} predictable"),
        ("every label-0 password reaches 10^14 guesses (Florencio et al.)",
         df[df.label == 0].password.map(log10_search_space).min() >= 14,
         f"min = 10^{df[df.label == 0].password.map(log10_search_space).min():.1f}"),
        ("every train row has a CV fold 0-4",
         train.fold.between(0, 4).all(), ""),
        ("unseen challenge words not in feature dictionary",
         not set(CHALLENGE_WORDS_UNSEEN) & set(COMMON_WORDS), ""),
    ]


def main():
    df = pd.read_csv(CLEAN_DATASET, keep_default_na=False,
                     dtype={"password": str, "family_id": str})
    results = run_checks(df)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n{len(failed)} check(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(results)} checks passed on {len(df):,} rows.")


if __name__ == "__main__":
    main()
