from src.preprocess import clean_text


def test_clean_text():
    message = "CONGRATULATIONS!!! You have WON $1000!!!"

    result = clean_text(message)

    assert result == "congratulations you have won 1000"


def test_extra_spaces():
    message = "Hello     World!!!"

    result = clean_text(message)

    assert result == "hello world"


def test_lowercase():
    message = "FREE MONEY NOW"

    result = clean_text(message)

    assert result == "free money now"