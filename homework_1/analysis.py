import pandas as pd
from collections import Counter
import string


def load_data(file_path):
    """
    Load the CSV data into a Pandas DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)


def compute_basic_statistics(data):
    """
    Compute basic statistics for the dataset.
    Args:
        data (pd.DataFrame): The SMS dataset.
    Returns:
        dict: Basic statistics.
    """
    num_messages = len(data)
    num_spams = sum(data["Category"] == "spam")
    total_word_count = data["Message"].str.split().str.len().sum()
    avg_words_per_message = round(total_word_count / num_messages, 3)
    return {
        "Number of SMS messages": num_messages,
        "Number of spam messages": num_spams,
        "Total word count": total_word_count,
        "Average words per message": avg_words_per_message,
    }


def compute_word_statistics(data):
    """
    Compute word-related statistics such as most frequent words and rare words.
    Args:
        data (pd.DataFrame): The SMS dataset.
    Returns:
        dict: Word statistics.
    """
    all_words = (
        data["Message"]
        .str.cat(sep=" ")
        .translate(str.maketrans("", "", string.punctuation))
        .lower()
        .split()
    )
    word_frequency = Counter(all_words)
    most_frequent_words = word_frequency.most_common(5)
    rare_words = [word for word, count in word_frequency.items() if count == 1]
    return {
        "5 most frequent words": most_frequent_words,
        "Number of rare words": len(rare_words),
    }


def analyze_spam_csv(data):
    """
    Analyze the spam CSV file and compute statistics.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        dict: Combined statistics for the dataset.
    """
    basic_stats = compute_basic_statistics(data)
    word_stats = compute_word_statistics(data)
    return {**basic_stats, **word_stats}


def compare_tokenizations(nltk_tokens, spacy_tokens):
    """
    Compare tokenized results from NLTK and spaCy.
    Args:
        nltk_tokens (list): Tokenized words using NLTK.
        spacy_tokens (list): Tokenized words using spaCy.
    Returns:
        dict: Differences in tokenization summarized as lengths.
    """
    nltk_counter = Counter(nltk_tokens)
    spacy_counter = Counter(spacy_tokens)

    nltk_unique = set(nltk_counter.keys()) - set(spacy_counter.keys())
    spacy_unique = set(spacy_counter.keys()) - set(nltk_counter.keys())
    common_tokens = set(nltk_counter.keys()).intersection(set(spacy_counter.keys()))

    return {
        'NLTK unique tokens count': len(nltk_unique),
        'spaCy unique tokens count': len(spacy_unique),
        'Common tokens count': len(common_tokens),
        'Total NLTK tokens': len(nltk_tokens),
        'Total spaCy tokens': len(spacy_tokens)
    }


def compare_lemmatizations(nltk_lemmas, spacy_lemmas):
    """
    Compare lemmatized results from NLTK and spaCy.
    Args:
        nltk_lemmas (list): Lemmatized words using NLTK.
        spacy_lemmas (list): Lemmatized words using spaCy.
    Returns:
        dict: Differences in lemmatization summarized as lengths.
    """
    nltk_counter = Counter(nltk_lemmas)
    spacy_counter = Counter(spacy_lemmas)

    nltk_unique = set(nltk_counter.keys()) - set(spacy_counter.keys())
    spacy_unique = set(spacy_counter.keys()) - set(nltk_counter.keys())
    common_lemmas = set(nltk_counter.keys()).intersection(set(spacy_counter.keys()))

    return {
        'NLTK unique lemmas count': len(nltk_unique),
        'spaCy unique lemmas count': len(spacy_unique),
        'Common lemmas count': len(common_lemmas),
        'Total NLTK lemmas': len(nltk_lemmas),
        'Total spaCy lemmas': len(spacy_lemmas)
    }


def compare_stemmings(nltk_stems, spacy_stems):
    """
    Compare stemmed results from NLTK and spaCy.
    Args:
        nltk_stems (list): Stemmed words using NLTK.
        spacy_stems (list): Stemmed words using spaCy.
    Returns:
        dict: Differences in stemming summarized as lengths.
    """
    nltk_counter = Counter(nltk_stems)
    spacy_counter = Counter(spacy_stems)

    nltk_unique = set(nltk_counter.keys()) - set(spacy_counter.keys())
    spacy_unique = set(spacy_counter.keys()) - set(nltk_counter.keys())
    common_stems = set(nltk_counter.keys()).intersection(set(spacy_counter.keys()))

    return {
        'NLTK unique stems count': len(nltk_unique),
        'spaCy unique stems count': len(spacy_unique),
        'Common stems count': len(common_stems),
        'Total NLTK stems': len(nltk_stems),
        'Total spaCy stems': len(spacy_stems)
    }

# git remote add origin git@github-personal:RyanWri/afeka-nlp-homeworks.git