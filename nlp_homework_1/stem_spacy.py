import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def stem_spacy(messages):
    """
    Approximate stemming using spaCy's lemmatization.
    Args:
        messages (pd.Series): SMS messages.
    Returns:
        list: Stemmed words using spaCy.
    """
    all_stems = []
    for message in messages:
        doc = nlp(message)
        stems = [
            token.lemma_[:4] for token in doc
        ]  # Taking first 4 characters as a rough "stem"
        all_stems.extend(stems)
    return all_stems
