def find_message_with_stem_lemma_disparity(messages, nltk_stems, nltk_lemmas):
    """
    Find a message where removing it reduces the number of stemmed tokens
    but does not affect the number of lemmatized tokens.

    Args:
        messages (list): List of original messages.
        nltk_stems (list): List of lists containing stemmed tokens for each message.
        nltk_lemmas (list): List of lists containing lemmatized tokens for each message.

    Returns:
        str: The first message meeting the condition, or None if no such message exists.
    """
    for idx, message in enumerate(messages):
        stems = set(nltk_stems[idx])
        lemmas = set(nltk_lemmas[idx])

        for token in stems:
            simulated_stems = stems - {token}
            if len(simulated_stems) < len(stems) and len(lemmas) == len(lemmas):
                return message  # Found a matching message
    return None  # No matching message found


def find_message_with_lemma_stem_disparity(messages, nltk_stems, nltk_lemmas):
    """
    Find a message where removing it reduces the number of lemmatized tokens
    but does not affect the number of stemmed tokens.

    Args:
        messages (list): List of original messages.
        nltk_stems (list): List of lists containing stemmed tokens for each message.
        nltk_lemmas (list): List of lists containing lemmatized tokens for each message.

    Returns:
        str: The first message meeting the condition, or None if no such message exists.
    """
    for idx, message in enumerate(messages):
        stems = set(nltk_stems[idx])
        lemmas = set(nltk_lemmas[idx])

        for token in lemmas:
            simulated_lemmas = lemmas - {token}
            if len(simulated_lemmas) < len(lemmas) and len(stems) == len(stems):
                return message  # Found a matching message
    return None  # No matching message found
