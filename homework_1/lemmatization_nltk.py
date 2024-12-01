import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def nltk_pos_tagger(nltk_tag):
    """
    Convert NLTK POS tags to WordNet POS tags for lemmatization.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_nltk(messages):
    """
    Lemmatize messages using NLTK.
    Args:
        messages (pd.Series): SMS messages.
    Returns:
        list: Lemmatized words using NLTK.
    """
    lemmatizer = WordNetLemmatizer()
    all_lemmas = []
    for message in messages:
        # Tokenize and POS tag the message
        tokens = nltk.word_tokenize(message)
        pos_tags = nltk.pos_tag(tokens)
        # Lemmatize each token
        lemmas = [lemmatizer.lemmatize(word, pos=nltk_pos_tagger(tag)) for word, tag in pos_tags]
        all_lemmas.extend(lemmas)
    return all_lemmas
