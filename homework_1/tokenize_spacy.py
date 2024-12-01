import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def tokenize_spacy(messages):
    """
    Tokenize messages using spaCy.
    Args:
        messages (pd.Series): SMS messages.
    Returns:
        list: Tokenized words using spaCy.
    """
    all_tokens = []
    for message in messages:
        doc = nlp(message)
        tokens = [token.text for token in doc]
        all_tokens.extend(tokens)
    return all_tokens
