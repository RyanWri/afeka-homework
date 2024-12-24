import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_syntactic_relationships(word_pairs, model):
    syntactic_similarities = {}
    for word1, word2 in word_pairs:
        similarity = model.wv.similarity(word1, word2)
        syntactic_similarities[(word1, word2)] = similarity
    return syntactic_similarities


def evaluate_semantic_relationships(word_pairs, model):
    semantic_similarities = {}
    for word1, word2 in word_pairs:
        similarity = model.wv.similarity(word1, word2)
        semantic_similarities[(word1, word2)] = similarity
    return semantic_similarities


def analogy_score(model, word1, word2, word3, expected_word):
    try:
        predicted_word = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0][0]
        return predicted_word == expected_word
    except KeyError:
        return False


def evaluate_word_embeddings(model):
    syntactic_pairs = [('run', 'runs'), ('eat', 'eats'), ('cat', 'cats')]
    semantic_pairs = [('king', 'queen'), ('man', 'woman'), ('apple', 'orange')]

    syntactic_results = evaluate_syntactic_relationships(syntactic_pairs, model)
    semantic_results = evaluate_semantic_relationships(semantic_pairs, model)

    analogy_results = [
        analogy_score(model, 'man', 'woman', 'king', 'queen'),
        analogy_score(model, 'cat', 'cats', 'run', 'running')
    ]

    return syntactic_results, semantic_results, analogy_results
