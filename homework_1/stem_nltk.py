from nltk.stem import PorterStemmer, LancasterStemmer
import nltk

# Ensure required NLTK data is downloaded
nltk.download("punkt")


def stem_nltk(messages, method="porter"):
    """
    Stem messages using NLTK.
    Args:
        messages (pd.Series): SMS messages.
        method (str): Stemming method ('porter' or 'lancaster').
    Returns:
        list: Stemmed words using NLTK.
    """
    if method == "porter":
        stemmer = PorterStemmer()
    elif method == "lancaster":
        stemmer = LancasterStemmer()
    else:
        raise ValueError("Invalid method. Choose 'porter' or 'lancaster'.")

    all_stems = []
    for message in messages:
        tokens = nltk.word_tokenize(message)
        stems = [stemmer.stem(word) for word in tokens]
        all_stems.extend(stems)
    return all_stems
