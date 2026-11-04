"""
STEP 4 - Password Feature Extraction
------------------------------------
Extracts numerical features from passwords for:
1. Rule-Based classification (System A)
2. Machine Learning models (System B)

Input:
    data/processed/passwords_clean.csv

Output:
    data/processed/password_features.csv

The original password is NOT included in the output.
Existing labels and other columns are preserved.

Run: python -m src.features.password_features
"""

import math
import re
from collections import Counter

import pandas as pd

from src.utils.common import CLEAN_DATASET, FEATURES_DATASET, LEET
from src.utils.wordlists import COMMON_WORDS, KEYBOARD_ROWS


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = CLEAN_DATASET
OUTPUT_PATH = FEATURES_DATASET

PASSWORD_COLUMN = "password"

MIN_WORD_LEN = 4   # ignore dictionary words shorter than this (too many chance matches)
DICTIONARY = [w for w in COMMON_WORDS if len(w) >= MIN_WORD_LEN]

YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)")   # 1950-2039


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def char_class(c):
    if c.islower():
        return "L"
    if c.isupper():
        return "U"
    if c.isdigit():
        return "D"
    return "S"


def longest_run(password, step):
    """Longest run of consecutive characters increasing (step=1) or
    decreasing (step=-1) by one code point: abcd, 1234, 4321."""
    best = run = 1 if password else 0
    for a, b in zip(password, password[1:]):
        run = run + 1 if ord(b) - ord(a) == step else 1
        best = max(best, run)
    return best


def has_keyboard_walk(normalized, min_len=4):
    """True if a substring of length >= min_len follows a keyboard row
    forwards or backwards (qwer, asdf, 7890, lkjh...)."""
    for row in KEYBOARD_ROWS:
        for seq in (row, row[::-1]):
            for i in range(len(seq) - min_len + 1):
                if seq[i:i + min_len] in normalized:
                    return True
    return False


def shannon_entropy(password):
    """Bits per character of the password's own character distribution."""
    if not password:
        return 0.0
    counts = Counter(password)
    n = len(password)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(password):
    """
    Extract numerical characteristics from one password.
    """

    if pd.isna(password):
        password = ""

    password = str(password)
    length = len(password)

    # 1. Character composition
    lower_count = sum(c.islower() for c in password)
    upper_count = sum(c.isupper() for c in password)
    digit_count = sum(c.isdigit() for c in password)
    whitespace_count = sum(c.isspace() for c in password)
    symbol_count = sum(not c.isalnum() and not c.isspace() for c in password)

    # 2. Character presence
    has_lower = int(lower_count > 0)
    has_upper = int(upper_count > 0)
    has_digits = int(digit_count > 0)
    has_symbols = int(symbol_count > 0)
    num_char_classes = has_lower + has_upper + has_digits + has_symbols

    # 3. Character diversity
    unique_chars = len(set(password))
    unique_ratio = unique_chars / length if length > 0 else 0
    repeated_chars = length - unique_chars
    entropy_per_char = shannon_entropy(password)

    # 4. Consecutive repeated characters (aaa)
    max_repeated_sequence = 0
    current_sequence = 0
    previous_char = None
    for char in password:
        current_sequence = current_sequence + 1 if char == previous_char else 1
        max_repeated_sequence = max(max_repeated_sequence, current_sequence)
        previous_char = char

    # 5. Sequences and keyboard patterns
    normalized_password = password.casefold()
    max_ascending_run = longest_run(normalized_password, 1)    # abcd, 1234
    max_descending_run = longest_run(normalized_password, -1)  # dcba, 4321
    has_sequence = int(max(max_ascending_run, max_descending_run) >= 3)
    keyboard_walk = int(has_keyboard_walk(normalized_password))

    # 6. Common words (after undoing l33t: P4ssw0rd -> password)
    deleet = normalized_password.translate(LEET)
    matched = [w for w in DICTIONARY if w in normalized_password or w in deleet]
    contains_common_word = int(bool(matched))
    longest_word = max((len(w) for w in matched), default=0)
    common_word_ratio = longest_word / length if length > 0 else 0
    leet_used = int(any(w in deleet and w not in normalized_password for w in matched))

    # 7. Years and dates
    contains_year = int(bool(YEAR_RE.search(password)))

    # 8. Structure: the typical human template Word + digits + symbol
    pattern = "".join(char_class(c) for c in password)          # Maria2026! -> ULLLLDDDDS
    segments = re.sub(r"(.)\1+", r"\1", pattern)                  # -> ULDS
    class_transitions = max(len(segments) - 1, 0)
    starts_with_upper = int(bool(password) and password[0].isupper()
                            and upper_count == 1)
    # digits at the end, ignoring trailing symbols: Maria2026! -> 4
    trailing_digits = len(re.search(r"(\d*)[^A-Za-z0-9]*$", password).group(1))
    ends_with_symbol = int(bool(password) and not password[-1].isalnum())
    letters_then_digits = int(bool(re.fullmatch(
        r"[A-Za-z]+[0-9]+[^A-Za-z0-9]*", password)))
    word_digits_symbol = int(bool(re.fullmatch(
        r"[A-Za-z]+[0-9]+[^A-Za-z0-9]+", password)))

    return {
        "length": length,

        "lower_count": lower_count,
        "upper_count": upper_count,
        "digit_count": digit_count,
        "symbol_count": symbol_count,
        "whitespace_count": whitespace_count,

        "has_lower": has_lower,
        "has_upper": has_upper,
        "has_digits": has_digits,
        "has_symbols": has_symbols,
        "num_char_classes": num_char_classes,

        "unique_chars": unique_chars,
        "unique_ratio": unique_ratio,
        "repeated_chars": repeated_chars,
        "entropy_per_char": entropy_per_char,

        "max_repeated_sequence": max_repeated_sequence,
        "max_ascending_run": max_ascending_run,
        "max_descending_run": max_descending_run,
        "has_sequence": has_sequence,
        "keyboard_walk": keyboard_walk,

        "contains_common_word": contains_common_word,
        "common_word_ratio": common_word_ratio,
        "leet_used": leet_used,
        "contains_year": contains_year,

        "class_transitions": class_transitions,
        "starts_with_upper": starts_with_upper,
        "trailing_digits": trailing_digits,
        "ends_with_symbol": ends_with_symbol,
        "letters_then_digits": letters_then_digits,
        "word_digits_symbol": word_digits_symbol,
    }


# --------------------------------------------------
# DATASET PROCESSING
# --------------------------------------------------

def process_dataset():

    print("Loading cleaned dataset...")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype={PASSWORD_COLUMN: "string"},
                     keep_default_na=False)
    if PASSWORD_COLUMN not in df.columns:
        raise ValueError(f"Column '{PASSWORD_COLUMN}' not found. "
                         f"Available columns: {list(df.columns)}")
    print(f"Loaded {len(df)} passwords.")

    print("Extracting password features...")
    features = pd.DataFrame(df[PASSWORD_COLUMN].map(extract_features).tolist(),
                            index=df.index)

    # Preserve labels and other original columns, but never the raw password.
    final_df = pd.concat([df.drop(columns=[PASSWORD_COLUMN]), features], axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction completed!")
    print(f"Total records: {len(final_df)}")
    print(f"Total features: {len(features.columns)}")
    print(f"Output saved to: {OUTPUT_PATH}")
    print("\nExtracted features:")
    print(list(features.columns))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    process_dataset()
