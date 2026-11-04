"""
Unit tests for src/features/password_features.py and src/utils/common.py.
Run from the repository root:  python -m pytest -q
"""
from src.features.password_features import extract_features
from src.utils.common import family_id, is_rule_compliant


def test_basic_counts():
    f = extract_features("Maria2026!")
    assert f["length"] == 10
    assert (f["upper_count"], f["lower_count"], f["digit_count"], f["symbol_count"]) == (1, 4, 4, 1)
    assert f["num_char_classes"] == 4


def test_predictable_template_is_detected():
    f = extract_features("Summer2026!")
    assert f["contains_common_word"] == 1
    assert f["contains_year"] == 1
    assert f["starts_with_upper"] == 1
    assert f["word_digits_symbol"] == 1
    assert f["trailing_digits"] == 4


def test_leet_word_is_detected():
    f = extract_features("P4ssw0rd1!")
    assert f["contains_common_word"] == 1
    assert f["leet_used"] == 1


def test_sequences_and_keyboard():
    assert extract_features("abc12345")["max_ascending_run"] == 5
    assert extract_features("qwerty")["keyboard_walk"] == 1
    assert extract_features("aaaa1")["max_repeated_sequence"] == 4


def test_random_password_has_no_patterns():
    f = extract_features("xK9#mQ2vLp8$")
    assert f["contains_common_word"] == 0
    assert f["contains_year"] == 0
    assert f["keyboard_walk"] == 0


def test_year_not_matched_inside_long_numbers():
    assert extract_features("abc120265")["contains_year"] == 0


def test_empty_password_does_not_crash():
    assert extract_features("")["length"] == 0


def test_family_id_groups_variants():
    assert family_id("Maria2026!") == family_id("maria2027#") == family_id("M4ria") == "maria"
    assert family_id("password1") == family_id("1password") == "password"
    assert family_id("123456!") == "num:123456"


def test_rule_compliance():
    assert is_rule_compliant("Summer2026!")
    assert not is_rule_compliant("summer2026")
