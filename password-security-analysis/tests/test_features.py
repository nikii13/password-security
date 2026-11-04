from src.features.password_features import extract_password_features

def test_feature_extraction():
    features = extract_password_features("Abc123!!")
    assert features["length"] == 8
    assert features["has_lower"] == 1
    assert features["has_upper"] == 1
    assert features["has_digit"] == 1
    assert features["has_symbol"] == 1

def test_empty_password():
    features = extract_password_features("")
    assert features["length"] == 0
    assert features["unique_char_ratio"] == 0
