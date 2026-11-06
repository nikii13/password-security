"""
PART 6 - Evaluation of the similarity module
============================================
1. Examples: the cases from the project plan plus typical user edits
   -> reports/tables/similarity_examples.csv
2. Experiment on real data: does the score separate variants of the same
   base password from unrelated passwords?
     - variant pairs:   two different RockYou passwords from the same family
                        (e.g. princess1 / Princess12)
     - unrelated pairs: two RockYou passwords from different families
   -> reports/tables/similarity_evaluation.csv
   -> reports/figures/fig7_similarity_distribution.png

Note: RockYou has no user names, so a variant pair is two people choosing
variations of the same base word, not one person reusing a password. It tests
whether the score recognises the transformations; the reuse figures come
from Das et al. (2014).

Input: data/samples/rockyou_sample.csv
Run:   python -m src.similarity.evaluate_similarity
"""
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.similarity.password_similarity import compare
from src.utils.common import REPORTS_DIR, ROCKYOU_SAMPLE, SEED

N_PAIRS = 3000
TABLES = REPORTS_DIR / "tables"
FIGURES = REPORTS_DIR / "figures"

EXAMPLES = [   # (scenario, password A, password B)
    ("Plan example: Netflix vs Amazon", "Maria2026!", "Maria2026#"),
    ("Plan example: Netflix vs Instagram", "Maria2026!", "Maria2027!"),
    ("Plan example: Amazon vs Instagram", "Maria2026#", "Maria2027!"),
    ("Yearly update", "Summer2025!", "Summer2026!"),
    ("Case change only", "password1", "PASSWORD1"),
    ("l33t substitution", "Password1", "P@ssw0rd1"),
    ("Digits appended", "dragon", "dragon123"),
    ("Symbol appended", "Welcome1", "Welcome1!"),
    ("Same word, different suffix", "football10", "Football2024!"),
    ("Different passwords", "Maria2026!", "Reykjavik88"),
    ("Random vs human", "Maria2026!", "xK9#mQ2vLp8$"),
]


def examples_table():
    rows = []
    for scenario, a, b in EXAMPLES:
        r = compare(a, b)
        rows.append({"scenario": scenario, "password_a": a, "password_b": b,
                     "similarity": r["similarity"], "edit_distance": r["edit_distance"],
                     "same_base": r["same_base"], "verdict": r["verdict"],
                     "changes": "; ".join(r["changes"])})
    return pd.DataFrame(rows)


def build_pairs(df, rng):
    words = df[~df.family_id.str.startswith("num:") & (df.family_id.str.len() >= 4)]
    groups = [g.password.tolist() for _, g in words.groupby("family_id") if g.password.nunique() >= 2]
    variant = []
    for _ in range(N_PAIRS):
        a, b = rng.sample(rng.choice(groups), 2)
        variant.append((a, b))
    pool = words[["password", "family_id"]].values.tolist()
    unrelated = []
    while len(unrelated) < N_PAIRS:
        (a, fa), (b, fb) = rng.sample(pool, 2)
        if fa != fb:
            unrelated.append((a, b))
    return variant, unrelated, len(groups)


def main():
    ex = examples_table()
    TABLES.mkdir(parents=True, exist_ok=True)
    ex.to_csv(TABLES / "similarity_examples.csv", index=False)
    pd.set_option("display.width", 180)
    print("Examples:")
    print(ex[["scenario", "password_a", "password_b", "similarity", "verdict", "changes"]].to_string(index=False))

    rng = random.Random(SEED)
    df = pd.read_csv(ROCKYOU_SAMPLE, keep_default_na=False, dtype=str)
    variant, unrelated, n_families = build_pairs(df, rng)
    results = pd.DataFrame(
        [{"pair_type": "variant (same family)", **compare(a, b)} for a, b in variant]
        + [{"pair_type": "unrelated", **compare(a, b)} for a, b in unrelated])

    summary = (results.groupby("pair_type")
               .agg(pairs=("similarity", "size"),
                    mean_similarity=("similarity", "mean"),
                    median_similarity=("similarity", "median"),
                    pct_near_duplicate=("verdict", lambda v: 100 * (v == "near-duplicate").mean()),
                    pct_related=("verdict", lambda v: 100 * (v == "related").mean()),
                    pct_different=("verdict", lambda v: 100 * (v == "different").mean()))
               .reset_index())
    # Score-only rates: variant pairs are DEFINED by sharing a family, and the
    # verdict also uses the family, so the verdict is circular for this test.
    # The similarity score alone is the fair measure.
    score = results.groupby("pair_type").similarity
    summary["pct_score_ge_0.60"] = summary.pair_type.map(score.apply(lambda s: 100 * (s >= 0.60).mean()))
    summary["pct_score_ge_0.80"] = summary.pair_type.map(score.apply(lambda s: 100 * (s >= 0.80).mean()))
    summary.round(2).to_csv(TABLES / "similarity_evaluation.csv", index=False)
    print(f"\nExperiment on RockYou ({n_families:,} families with 2+ variants, {N_PAIRS:,} pairs per type):")
    print(summary.round(2).to_string(index=False))

    # figure
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#e4e3df", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.labelcolor": "#52514e"})
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bins = np.linspace(0, 1, 26)
    for label, color in [("unrelated", "#eb6834"), ("variant (same family)", "#2a78d6")]:
        vals = results.loc[results.pair_type == label, "similarity"]
        ax.hist(vals, bins=bins, weights=np.full(len(vals), 100 / len(vals)),
                histtype="step", linewidth=2, color=color,
                label="Unrelated passwords" if label == "unrelated" else "Variants of the same base word")
    for x, txt in [(0.6, "related ≥ 0.60"), (0.8, "near-duplicate ≥ 0.80")]:
        ax.axvline(x, color="#52514e", linestyle=(0, (4, 3)), linewidth=1)
        ax.text(x + 0.005, ax.get_ylim()[1] * 0.93, txt, fontsize=9, color="#52514e")
    ax.set_xlabel("Similarity score (1 = same after normalising case and l33t)")
    ax.set_ylabel("% of pairs")
    ax.yaxis.grid(True, color="#e4e3df")
    ax.set_axisbelow(True)
    ax.set_title("Similarity of variant vs unrelated RockYou password pairs", loc="left",
                 fontweight="bold", fontsize=13, pad=26)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=10)
    fig.savefig(FIGURES / "fig7_similarity_distribution.png", dpi=200, bbox_inches="tight")
    print("  saved reports/figures/fig7_similarity_distribution.png")


if __name__ == "__main__":
    main()
