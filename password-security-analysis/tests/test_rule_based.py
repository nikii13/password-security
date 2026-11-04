from src.models.rule_based import classify_password

def test_rule_based_returns_known_class():
    assert classify_password("abc") in {"weak", "medium", "strong"}
