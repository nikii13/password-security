# Password Security Analysis

Can a lightweight machine-learning model identify weak and predictable
passwords that traditional rule-based checkers classify as secure?

## Pipeline (run in this order, from the repository root)

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| 1. Build samples | `python -m src.data.sample_dataset` | `data/raw/rockyou.txt` | `data/samples/{rockyou,random,challenge}_sample.csv` |
| 2. Clean, label, split | `python -m src.data.clean_dataset` | the 3 samples | `data/processed/passwords_clean.csv` |
| 3. Validate | `python -m src.data.validate_dataset` | `passwords_clean.csv` | pass/fail report (stops on failure) |
| 4. Extract features | `python -m src.features.password_features` | `passwords_clean.csv` | `data/processed/password_features.csv` |
| 5. System A | `src/models/rule_based.py` | features | *(next part)* |
| 6. System B | `src/models/train_ml.py` | features (`split`, `fold`) | *(next part)* |

Tests: `python -m pytest -q`

With Docker (no local Python needed):

```
docker compose run --rm pipeline
docker compose run --rm tests
```

## Data

- `data/raw/rockyou.txt`: the 2009 RockYou leak. **Never committed.**
  Place it here manually before step 1.
- `data/samples/`: the three source datasets.
  - **A.** `rockyou_sample.csv`: 99,997 human passwords, stratified by frequency rank, plus 20,000 rule-compliant ones.
  - **B.** `random_sample.csv`: 100,000 passwords from Python's `secrets`, with varied composition.
  - **C.** `challenge_sample.csv`: about 1,160 passwords that pass LUDS rules but follow attacker templates.
- `data/processed/`: generated; rebuild with the scripts.
- `reports/tables/`: row counts per step and split summary, for the report.

## Labels (operational)

| label | label_name | sources |
| --- | --- | --- |
| 1 | predictable | RockYou, challenge set |
| 0 | not_predictable | secure random generator |

Labels come from the **source** of each password, never from the rule-based
checker, so the ML model cannot simply learn to copy System A.

Limitations:
- A rare RockYou password is not necessarily weak.
- A random password is not strong in every context, e.g. if it is short or reused.

## Splits

| split | content | use |
| --- | --- | --- |
| `train` (80%) | RockYou + random, with `fold` 0–4 | training and cross-validation |
| `test` (20%) | RockYou + random | final evaluation, used once |
| `challenge` | challenge set only | never trained on; tests the research question |

Splitting is by `family_id` (Maria2026!, maria2027 and M4ria share the family
`maria`), so no variant of a password appears in both train and test.

The challenge vocabulary is half `seen` (in the feature dictionary) and half
`unseen` (not in it). Report results for both halves.
