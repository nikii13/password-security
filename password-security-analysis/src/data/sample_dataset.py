"""
STEP 1 - Build the three samples
================================
Creates the three source datasets of the project in data/samples/:

  A. rockyou_sample.csv    human-chosen passwords, stratified by frequency
  B. random_sample.csv     passwords from a cryptographically secure generator
  C. challenge_sample.csv  passwords that PASS traditional rules but are
                           predictable (Summer2026!, Maria26#...)

Also writes the tables for the report to reports/tables/.

Run from the repository root:
    python -m src.data.sample_dataset               # all three
    python -m src.data.sample_dataset --only random # just one
"""
import argparse
import csv
import random
import secrets
import string

from src.utils.common import (CHALLENGE_SAMPLE, RANDOM_SAMPLE, REPORTS_DIR,
                              ROCKYOU_PATH, ROCKYOU_SAMPLE, SAMPLES_DIR, SEED,
                              family_id, is_rule_compliant, is_valid_password)
from src.utils.wordlists import CHALLENGE_WORDS_SEEN, CHALLENGE_WORDS_UNSEEN

TABLES_DIR = REPORTS_DIR / "tables"
COLUMNS = ["password", "stratum", "vocab", "rule_compliant", "family_id"]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ===========================================================================
# A. RockYou: stratified sample
# ===========================================================================
# rockyou.txt has no counts but is sorted from most to least frequent,
# so the line number (rank) tells us how common a password is.
# (stratum, first rank, last rank (None = end), sample size (None = all))
ROCKYOU_BANDS = [
    ("very_common", 1, 10_000, None),
    ("common", 10_001, 1_000_000, 30_000),
    ("rare", 1_000_001, None, 40_000),
]
# Extra LUDS-compliant passwords: without them almost no "human" password has
# a symbol and the model would only learn "symbol = strong" (the LUDS rule).
ROCKYOU_RULE_COMPLIANT_N = 20_000


def sample_rockyou(rng):
    log, seen, rows = [], set(), []
    raw = decoded = valid = 0
    with open(ROCKYOU_PATH, "rb") as f:
        for rank, line in enumerate(f, start=1):
            raw += 1
            try:
                pw = line.rstrip(b"\n").rstrip(b"\r").decode("utf-8")
            except UnicodeDecodeError:
                continue
            decoded += 1
            if not is_valid_password(pw):
                continue
            valid += 1
            if pw not in seen:
                seen.add(pw)
                rows.append((rank, pw))
    log += [("raw_lines", raw), ("valid_utf8", decoded),
            ("valid_chars_and_length", valid), ("after_exact_dedup", len(rows))]

    chosen, summary = {}, []                      # password -> (rank, stratum)
    max_rank = rows[-1][0]
    for name, lo, hi, n in ROCKYOU_BANDS:
        hi = hi or max_rank
        pool = [(r, p) for r, p in rows if lo <= r <= hi]
        pick = pool if n is None or n >= len(pool) else rng.sample(pool, n)
        chosen.update({p: (r, name) for r, p in pick})
        summary.append((name, f"{lo}-{hi}", len(pool), len(pick)))

    pool = [(r, p) for r, p in rows if is_rule_compliant(p) and p not in chosen]
    n = min(ROCKYOU_RULE_COMPLIANT_N, len(pool))
    chosen.update({p: (r, "rule_compliant_extra") for r, p in rng.sample(pool, n)})
    summary.append(("rule_compliant_extra", "any", len(pool), n))
    log.append(("final_sample", len(chosen)))

    out = [(p, s, "", int(is_rule_compliant(p)), family_id(p))
           for p, (r, s) in sorted(chosen.items(), key=lambda kv: kv[1][0])]
    write_csv(ROCKYOU_SAMPLE, COLUMNS, out)
    write_csv(TABLES_DIR / "rockyou_cleaning_log.csv", ["step", "rows"], log)
    write_csv(TABLES_DIR / "rockyou_strata.csv",
              ["stratum", "rank_range", "pool_size", "sampled"], summary)
    return len(out)


# ===========================================================================
# B. Random passwords (secure generator)
# ===========================================================================
# `secrets` uses the OS's cryptographically secure generator (`random` is not
# suitable for passwords). The seed only fixes the recipe (lengths, mix);
# the passwords themselves differ on every run, as random passwords should.
RANDOM_N = 100_000                 # same size as RockYou -> balanced classes
RANDOM_MIN_LEN, RANDOM_MAX_LEN = 8, 16
LOWER, UPPER = string.ascii_lowercase, string.ascii_uppercase
DIGITS, SYMBOLS = string.digits, "!@#$%^&*()-_=+[]{};:,.?/"
# Varied composition so that "strong" does not always mean "has a symbol"
RANDOM_COMPOSITIONS = {            # name -> (pools that must appear, share)
    "all_four_classes": ([LOWER, UPPER, DIGITS, SYMBOLS], 0.40),
    "lower_upper_digit": ([LOWER, UPPER, DIGITS], 0.20),
    "lower_digit": ([LOWER, DIGITS], 0.20),
    "lower_only": ([LOWER], 0.20),
}


def make_random_password(pools, length):
    """At least one char from each pool, the rest from their union, shuffled."""
    alphabet = "".join(pools)
    chars = [secrets.choice(p) for p in pools]
    chars += [secrets.choice(alphabet) for _ in range(length - len(chars))]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def sample_random(rng):
    rows, seen = [], set()
    for name, (pools, share) in RANDOM_COMPOSITIONS.items():
        made = 0
        while made < round(RANDOM_N * share):
            pw = make_random_password(pools, rng.randint(RANDOM_MIN_LEN, RANDOM_MAX_LEN))
            if pw in seen:
                continue
            seen.add(pw)
            rows.append((pw, name, "", int(is_rule_compliant(pw)), family_id(pw)))
            made += 1
    write_csv(RANDOM_SAMPLE, COLUMNS, rows)
    return len(rows)


# ===========================================================================
# C. Challenge set
# ===========================================================================
# Rule-compliant passwords built from templates attackers try early.
# NEVER used for training. Half the words are deliberately NOT in the
# feature dictionary ("unseen") to test structure, not memorised words.
CHALLENGE_PER_WORD = 40
CHALLENGE_SYMBOLS = list("!@#$*.?")
LEET_FORWARD = {"a": "4", "e": "3", "o": "0", "i": "1", "s": "5"}


def make_challenge_password(word, rng):
    w = word.capitalize()
    year = str(rng.randint(1960, 2030))
    sym = rng.choice(CHALLENGE_SYMBOLS)
    leet = "".join(LEET_FORWARD[c] if c in LEET_FORWARD and rng.random() < 0.6
                   else c for c in w)
    return rng.choice([
        ("word_year_symbol", f"{w}{year}{sym}"),                       # Summer2026!
        ("word_2digit_symbol", f"{w}{year[2:]}{sym}"),                 # Maria26#
        ("word_123_symbol", f"{w}{rng.choice(['1', '12', '123', '1234'])}{sym}"),
        ("word_symbol_year", f"{w}{sym}{year}"),                       # Madrid!2026
        ("leet_word_digit_symbol", f"{leet}{rng.randint(1, 99)}{sym}"),  # P4ssw0rd7!
    ])


def sample_challenge(rng):
    rows, seen = [], set()
    for vocab, words in (("seen", CHALLENGE_WORDS_SEEN),
                         ("unseen", CHALLENGE_WORDS_UNSEEN)):
        for word in words:
            for _ in range(CHALLENGE_PER_WORD):
                template, pw = make_challenge_password(word, rng)
                # every challenge item must pass the traditional rules
                if pw in seen or not is_rule_compliant(pw):
                    continue
                seen.add(pw)
                rows.append((pw, template, vocab, 1, family_id(pw)))
    write_csv(CHALLENGE_SAMPLE, COLUMNS, rows)
    return len(rows)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rockyou", "random", "challenge"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    steps = {"rockyou": sample_rockyou, "random": sample_random,
             "challenge": sample_challenge}
    for name, fn in steps.items():
        if args.only in (None, name):
            n = fn(random.Random(args.seed))
            print(f"[{name}] {n:,} passwords saved")


if __name__ == "__main__":
    main()
