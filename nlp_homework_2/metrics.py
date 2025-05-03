from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarity(word, model1, model2):
    vector1 = model1.wv[word].reshape(1, -1)
    vector2 = model2.wv[word].reshape(1, -1)
    return cosine_similarity(vector1, vector2)[0][0]


# Compare syntactic relationships using both models
def compare_syntactic_relationships(syntactic_pairs, skipgram_model, cbow_model):
    syntactic_results = {}
    for word1, word2 in syntactic_pairs:
        similarity_skipgram = compute_cosine_similarity(
            word1, skipgram_model, skipgram_model
        )
        similarity_cbow = compute_cosine_similarity(word1, cbow_model, cbow_model)
        syntactic_results[(word1, word2)] = {
            "skipgram": similarity_skipgram,
            "cbow": similarity_cbow,
        }
    return syntactic_results
