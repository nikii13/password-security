"""
PART 6 - Password similarity detection
======================================
Detects passwords that are not identical but are small variations of each
other, e.g. one per website:

    Netflix   Maria2026!
    Amazon    Maria2026#      <- symbol changed
    Instagram Maria2027!      <- year +1

Why it matters: changing one digit or symbol does not protect you. Users
modify passwords with a small set of predictable rules, and attackers who
know one password guess its variants in very few tries (Das et al., 2014:
30% of modified passwords guessed within 100 attempts).

This module is independent of the ML classifier: it does not use the model,
only the two passwords.

How the similarity score is computed:
  1. Normalise both passwords: lowercase + undo l33t (P4ssw0rd -> password),
     so cosmetic changes do not hide the similarity.
  2. Levenshtein edit distance on the normalised strings: the minimum number
     of single-character insertions, deletions or substitutions.
  3. similarity = 1 - distance / length of the longer password  (0 to 1).
  4. Extra signals: same base word (family), same structure (e.g. ULLLLDDDDS).

Verdict:
    identical       exactly the same password
    near-duplicate  same base word and <= 3 edits, or similarity >= 0.80
    related         same base word, or similarity >= 0.60
    different       anything else

Use from Python:
    from src.similarity.password_similarity import compare
    compare("Maria2026!", "Maria2027!")

Use from the terminal (compares every pair):
    python -m src.similarity.password_similarity Maria2026! Maria2026# Maria2027!
"""
import difflib
import re
import sys
from itertools import combinations

from src.utils.common import LEET, family_id

NEAR_DUPLICATE_SIMILARITY = 0.80
RELATED_SIMILARITY = 0.60
NEAR_DUPLICATE_MAX_EDITS = 3


# --------------------------------------------------
# BUILDING BLOCKS
# --------------------------------------------------

def normalize(password):
    """Lowercase and undo l33t inside words (P4ssw0rd! -> password!).
    Digits at the ends are kept, since they are usually years or counters."""
    pw = str(password)
    core = re.match(r"^([^A-Za-z]*)(.*?)([^A-Za-z]*)$", pw)
    start, middle, end = core.groups()
    return start + middle.lower().translate(LEET) + end.lower()


def levenshtein(a, b):
    """Minimum number of insertions, deletions or substitutions to turn a into b."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,                 # deletion
                               current[j - 1] + 1,              # insertion
                               previous[j - 1] + (ca != cb)))   # substitution
        previous = current
    return previous[-1]


def structure(password):
    """Character-class pattern: Maria2026! -> ULLLLDDDDS."""
    return "".join("U" if c.isupper() else "L" if c.islower() else "D" if c.isdigit() else "S"
                   for c in str(password))


def describe_changes(a, b):
    """Human-readable list of what changed from a to b."""
    if a == b:
        return []
    changes = []
    if a.lower() == b.lower():
        return ["only uppercase/lowercase changed"]
    if normalize(a) == normalize(b):
        return ["only l33t substitutions or case changed"]

    num_a, num_b = re.findall(r"\d+", a), re.findall(r"\d+", b)
    if (len(num_a) == len(num_b) == 1 and num_a[0] != num_b[0]
            and re.sub(r"\d+", "#", a) == re.sub(r"\d+", "#", b)):
        diff = int(num_b[0]) - int(num_a[0])
        if abs(diff) <= 5:
            return [f"number {'incremented' if diff > 0 else 'decremented'} by {abs(diff)} "
                    f"({num_a[0]} -> {num_b[0]})"]
        return [f"number changed ({num_a[0]} -> {num_b[0]})"]

    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        old, new = a[i1:i2], b[j1:j2]
        where = "at the end" if i1 >= len(a) - max(1, len(old)) else (
            "at the start" if i1 == 0 else "in the middle")
        if op == "replace":
            changes.append(f"'{old}' replaced by '{new}' {where}")
        elif op == "insert":
            changes.append(f"'{new}' added {where}")
        elif op == "delete":
            changes.append(f"'{old}' removed {where}")
    return changes


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def compare(a, b):
    """
    Compare two passwords. Returns a dict with:
        similarity      0 (nothing in common) to 1 (same after normalising)
        edit_distance   edits between the normalised passwords
        raw_distance    edits between the passwords as typed
        same_base       both share the same base word (family)
        same_structure  same character-class pattern
        verdict         identical / near-duplicate / related / different
        changes         what changed, in words
    """
    a, b = str(a), str(b)
    na, nb = normalize(a), normalize(b)
    distance = levenshtein(na, nb)
    longest = max(len(na), len(nb), 1)
    similarity = 1 - distance / longest
    same_base = family_id(a) == family_id(b) and not family_id(a).startswith("num:")

    if a == b:
        verdict = "identical"
    elif (same_base and distance <= NEAR_DUPLICATE_MAX_EDITS) or similarity >= NEAR_DUPLICATE_SIMILARITY:
        verdict = "near-duplicate"
    elif same_base or similarity >= RELATED_SIMILARITY:
        verdict = "related"
    else:
        verdict = "different"

    return {
        "password_a": a,
        "password_b": b,
        "similarity": round(similarity, 3),
        "edit_distance": distance,
        "raw_distance": levenshtein(a, b),
        "same_base": same_base,
        "same_structure": structure(a) == structure(b),
        "verdict": verdict,
        # listing edits between unrelated passwords is noise, so only for similar ones
        "changes": describe_changes(a, b) if verdict != "different" else [],
    }


def compare_all(passwords):
    """Compare every pair in a list (e.g. one password per website)."""
    return [compare(a, b) for a, b in combinations(passwords, 2)]


def warning(result):
    """Short message for the user interface (Part 7)."""
    return {
        "identical": "Same password reused: one leak exposes both accounts.",
        "near-duplicate": "Almost the same password: an attacker who knows one will guess the other in a few tries.",
        "related": "Passwords share a base word or a large part: still risky if one leaks.",
        "different": "No meaningful similarity detected.",
    }[result["verdict"]]


def main(argv):
    passwords = argv or ["Maria2026!", "Maria2026#", "Maria2027!", "M4ria2026!", "xK9#mQ2vLp8$"]
    for r in compare_all(passwords):
        print(f"{r['password_a']:>14}  vs  {r['password_b']:<14}  similarity={r['similarity']:.2f}  "
              f"edits={r['edit_distance']}  {r['verdict']:<14} {'; '.join(r['changes'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
