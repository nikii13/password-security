"""
Shared paths and helpers used by every pipeline step.

Keeping these in one place guarantees that "valid password",
"rule-compliant" and "family" mean exactly the same thing in sampling,
cleaning, validation and feature extraction.
"""
import re
import unicodedata
from pathlib import Path

# ---- paths (repo root = two levels above this file) ----------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SAMPLES_DIR = ROOT / "data" / "samples"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

ROCKYOU_PATH = RAW_DIR / "rockyou.txt"
ROCKYOU_SAMPLE = SAMPLES_DIR / "rockyou_sample.csv"
RANDOM_SAMPLE = SAMPLES_DIR / "random_sample.csv"
CHALLENGE_SAMPLE = SAMPLES_DIR / "challenge_sample.csv"
CLEAN_DATASET = PROCESSED_DIR / "passwords_clean.csv"
FEATURES_DATASET = PROCESSED_DIR / "password_features.csv"

# ---- global settings ------------------------------------------------------
SEED = 42
MIN_LEN, MAX_LEN = 4, 64

# Undo common l33t substitutions (p4ssw0rd -> password)
LEET = str.maketrans({"4": "a", "@": "a", "3": "e", "1": "i", "!": "i",
                      "0": "o", "5": "s", "$": "s", "7": "t"})


def is_valid_password(pw) -> bool:
    """Cleaning rule shared by all sources: not empty, no control
    characters, length between MIN_LEN and MAX_LEN."""
    if not isinstance(pw, str) or not pw.strip():
        return False
    if any(unicodedata.category(c).startswith("C") for c in pw):
        return False
    return MIN_LEN <= len(pw) <= MAX_LEN


def is_rule_compliant(pw: str) -> bool:
    """Classic LUDS rule: >= 8 characters with lowercase, uppercase,
    digit and symbol. Used to build the samples; System A
    (src/models/rule_based.py) implements its own full scoring."""
    return (len(pw) >= 8
            and any(c.islower() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
            and any(not c.isalnum() for c in pw))


def family_id(pw: str) -> str:
    """Group variants of the same base password so they never end up
    split between train and test (data leakage).

        Maria2026!, maria2027#, M4ria  -> 'maria'
        password1, Passw0rd!, 1password -> 'password'
        123456, 123456!                 -> 'num:123456'
    """
    if not any(c.isalpha() for c in pw):
        digits = re.sub(r"\D", "", pw)
        return "num:" + (digits if digits else pw)
    # strip digits/symbols at both ends, then undo l33t inside the word
    core = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", pw)
    base = re.sub(r"[^a-z]", "", core.lower().translate(LEET))
    return base if base else "num:" + pw
