import pytest

from src.predict import load_model, predict_message


def test_model_loads():
    # Verify that our trained model loads successfully
    model = load_model()

    assert model is not None


def test_spam_prediction():
    # Test a message containing common spam language
    message = "Congratulations! Claim your free prize now!"

    result = predict_message(message)

    assert result["prediction"] in ["spam", "ham"]
    assert 0 <= result["confidence"] <= 100


def test_ham_prediction():
    # Test a normal message
    message = "Hey, are we still meeting for lunch?"

    result = predict_message(message)

    assert result["prediction"] in ["spam", "ham"]
    assert 0 <= result["confidence"] <= 100


def test_empty_message():
    # Empty messages should raise an error
    with pytest.raises(ValueError):
        predict_message("")


def test_whitespace_message():
    # Messages containing only spaces should raise an error
    with pytest.raises(ValueError):
        predict_message("   ")


def test_prediction_structure():
    # Verify that predictions return the expected information
    message = "You have won a free vacation!"

    result = predict_message(message)

    assert "message" in result
    assert "prediction" in result
    assert "confidence" in result

    assert result["message"] == message