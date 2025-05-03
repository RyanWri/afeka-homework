import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def lemmatize_spacy(messages):
    """
    Lemmatize messages using spaCy.
    Args:
        messages (pd.Series): SMS messages.
    Returns:
        list: Lemmatized words using spaCy.
    """
    all_lemmas = []
    for message in messages:
        doc = nlp(message)
        # Extract lemmas
        lemmas = [token.lemma_ for token in doc]
        all_lemmas.extend(lemmas)
    return all_lemmas
