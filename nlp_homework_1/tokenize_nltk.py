from nltk.tokenize import word_tokenize
import nltk

# Ensure required NLTK data is downloaded
nltk.download("punkt")
nltk.download("punkt_tab")


def tokenize_nltk(messages):
    """
    Tokenize messages using NLTK.
    Args:
        messages (pd.Series): SMS messages.
    Returns:
        list: Tokenized words using NLTK.
    """
    all_tokens = []
    for message in messages:
        tokens = word_tokenize(message)
        all_tokens.extend(tokens)
    return all_tokens
