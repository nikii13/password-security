"""
Unit tests for Part 6 (src/similarity/password_similarity.py).
Run from the repository root:  python -m pytest -q
"""
from src.similarity.password_similarity import (compare, compare_all, levenshtein,
                                                normalize, warning)


def test_levenshtein_basic():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3


def test_normalize_undoes_case_and_leet():
    assert normalize("P4ssw0rd1") == normalize("password1") == "password1"


def test_plan_examples_are_near_duplicates():
    """The three passwords from the project plan (Netflix / Amazon / Instagram)."""
    for r in compare_all(["Maria2026!", "Maria2026#", "Maria2027!"]):
        assert r["verdict"] == "near-duplicate", r
        assert r["same_base"]


def test_year_increment_is_described():
    r = compare("Summer2025!", "Summer2026!")
    assert "incremented by 1" in r["changes"][0]


def test_identical_and_different():
    assert compare("Maria2026!", "Maria2026!")["verdict"] == "identical"
    r = compare("Maria2026!", "xK9#mQ2vLp8$")
    assert r["verdict"] == "different"
    assert r["similarity"] < 0.3
    assert r["changes"] == []


def test_similarity_is_symmetric_and_bounded():
    a, b = compare("dragon", "dragon123"), compare("dragon123", "dragon")
    assert a["similarity"] == b["similarity"]
    assert 0 <= a["similarity"] <= 1


def test_warning_message_exists_for_every_verdict():
    for pair in [("a1b2c3", "a1b2c3"), ("Maria2026!", "Maria2027!"), ("abc", "xyz")]:
        assert warning(compare(*pair))
