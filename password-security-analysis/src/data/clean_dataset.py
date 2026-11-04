"""
STEP 2 - Clean, merge, label and split
======================================
Inputs:  data/samples/rockyou_sample.csv
         data/samples/random_sample.csv
         data/samples/challenge_sample.csv
Output:  data/processed/passwords_clean.csv   (input of password_features.py)
         reports/tables/cleaning_log.csv
         reports/tables/split_summary.csv

What it does:
  1. Applies the same validity rules to every source.
  2. Removes exact duplicates, also across sources (challenge wins, so a
     challenge password never also appears in training data).
  3. Adds the operational label (from the SOURCE, never from System A):
         label = 1 "predictable"      RockYou + challenge
         label = 0 "not_predictable"  secure random generator
  4. Adds `split` (train / test / challenge) and `fold` (0-4, for
     cross-validation inside train). Splitting is done here, before
     feature extraction, so every later step reads the same split.

Why group-aware splitting: Maria2026!, maria2027 and M4ria share the
family_id 'maria'. If one were in train and another in test, the test score
would be inflated (data leakage). StratifiedGroupKFold keeps every family
on one side and keeps label proportions similar.

Run: python -m src.data.clean_dataset
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.common import (CHALLENGE_SAMPLE, CLEAN_DATASET, RANDOM_SAMPLE,
                              REPORTS_DIR, ROCKYOU_SAMPLE, SEED, is_valid_password)

SOURCES = [   # order = priority when the same password appears twice
    ("challenge", CHALLENGE_SAMPLE, 1),
    ("rockyou", ROCKYOU_SAMPLE, 1),
    ("random", RANDOM_SAMPLE, 0),
]
LABEL_NAMES = {1: "predictable", 0: "not_predictable"}
N_FOLDS = 5   # 1 of 5 = 20% test, as in the project plan (80/20)
TABLES_DIR = REPORTS_DIR / "tables"


def load_and_clean(log):
    frames = []
    for name, path, label in SOURCES:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["source"], df["label"] = name, label
        log.append((f"loaded_{name}", len(df)))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    log.append(("merged", len(df)))

    df = df[df["password"].map(is_valid_password)]
    log.append(("after_validity_filter", len(df)))

    df = df.drop_duplicates(subset="password", keep="first")
    log.append(("after_cross_source_dedup", len(df)))

    # Informational for the report: RockYou rows whose family also appears
    # in the challenge set (kept on purpose: 'summer1' is a real human password)
    ch_fams = set(df.loc[df.source == "challenge", "family_id"])
    log.append(("info_rockyou_rows_sharing_a_challenge_family",
                int(((df.source == "rockyou") & df.family_id.isin(ch_fams)).sum())))

    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["rule_compliant"] = df["rule_compliant"].astype(int)
    return df.reset_index(drop=True)


def assign_splits(df):
    df["split"], df["fold"] = "challenge", -1
    pool = df.index[df.source != "challenge"]

    # 20% test (whole families only, label proportions kept)
    outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr, te = next(outer.split(pool, df.loc[pool, "label"], df.loc[pool, "family_id"]))
    train_idx, test_idx = pool[tr], pool[te]
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # cross-validation folds inside train
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, idx) in enumerate(inner.split(train_idx, df.loc[train_idx, "label"],
                                             df.loc[train_idx, "family_id"])):
        df.loc[train_idx[idx], "fold"] = k
    return df


def main():
    log = []
    df = assign_splits(load_and_clean(log))

    columns = ["password", "source", "stratum", "vocab", "label", "label_name",
               "rule_compliant", "family_id", "split", "fold"]
    CLEAN_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(CLEAN_DATASET, index=False)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log, columns=["step", "rows"]).to_csv(
        TABLES_DIR / "cleaning_log.csv", index=False)
    summary = df.groupby(["split", "source", "label_name"]).size().rename("rows")
    summary.reset_index().to_csv(TABLES_DIR / "split_summary.csv", index=False)

    for step, n in log:
        print(f"{step:48s} {n:>9,}")
    print("\n" + summary.to_string())
    print(f"\nSaved {len(df):,} rows to {CLEAN_DATASET}")


if __name__ == "__main__":
    main()
