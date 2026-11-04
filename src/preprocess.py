import re
import string


def clean_text(text):
    """
    Clean a text message before machine learning.

    Steps:
    1. Convert text to lowercase.
    2. Remove punctuation.
    3. Remove extra whitespace.
    """

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at the beginning and end
    text = text.strip()

    return text