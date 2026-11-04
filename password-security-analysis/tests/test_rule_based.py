"""
Unit tests for System A (src/models/rule_based.py).
Run from the repository root:  python -m pytest -q
"""
from src.models.rule_based import classify


def test_short_password_is_weak():
    assert classify("Ab1!")["strength"] == "Weak"


def test_two_classes_is_weak():
    assert classify("password123")["strength"] == "Weak"


def test_three_classes_is_medium():
    assert classify("Password123")["strength"] == "Medium"


def test_three_classes_long_is_strong():
    assert classify("Password12345")["strength"] == "Strong"


def test_four_classes_is_strong():
    assert classify("xK9#mQ2vLp8$")["strength"] == "Strong"


def test_predictable_but_compliant_is_strong():
    """The blind spot System A is expected to have (research question)."""
    result = classify("Summer2026!")
    assert result["strength"] == "Strong"
    assert result["prediction"] == 0


def test_reports_passed_and_failed_rules():
    result = classify("hello")
    assert "has_lowercase" in result["passed_rules"]
    assert "min_length_8" in result["failed_rules"]
    assert "has_symbol" in result["failed_rules"]


def test_binary_mapping():
    assert classify("123456")["prediction"] == 1          # Weak   -> predictable
    assert classify("Password123")["prediction"] == 1     # Medium -> predictable
    assert classify("xK9#mQ2vLp8$")["prediction"] == 0    # Strong -> not predictable


def test_empty_password_does_not_crash():
    assert classify("")["strength"] == "Weak"
