import gensim
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText


# Function to read sentences from a file
def read_sentences(file_path):
    with open(file_path, "r") as file:
        corpus = file.readlines()
    return [sentence.strip().lower().split() for sentence in corpus]


# Function to train Skip-gram and CBOW models with a specified window size
def train_word2vec_models(sentences, vector_size, window_size=5):
    # Train Skip-gram model
    skipgram_model = gensim.models.Word2Vec(
        sentences, vector_size=vector_size, window=window_size, min_count=1, sg=1
    )

    # Train CBOW model
    cbow_model = gensim.models.Word2Vec(
        sentences, vector_size=vector_size, window=window_size, min_count=1, sg=0
    )

    return skipgram_model, cbow_model


# Function to compute cosine similarity between word embeddings from both models
def compute_cosine_similarity(word, model1, model2):
    vector1 = model1.wv[word].reshape(1, -1)  # Skip-gram embedding
    vector2 = model2.wv[word].reshape(1, -1)  # CBOW embedding
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]


# Function to compare cosine similarities between Skip-gram and CBOW models for a list of words
def compare_models_for_words(words, skipgram_model, cbow_model):
    similarities = {}
    for word in words:
        similarity = compute_cosine_similarity(word, skipgram_model, cbow_model)
        similarities[word] = similarity
    return similarities


# Function to compute cosine similarity for words with semantic and syntactic relationships
def compare_syntactic_semantic_relationships(skipgram_model, cbow_model):
    # Semantic relationships (e.g., synonyms)
    semantic_pairs = [("king", "queen"), ("man", "woman"), ("apple", "orange")]

    # Syntactic relationships (e.g., analogy: man is to woman as king is to queen)
    syntactic_pairs = [("cat", "cats"), ("run", "running"), ("eat", "eats")]

    # Comparing semantic relationships
    print("Comparing Semantic Relationships:")
    for word1, word2 in semantic_pairs:
        similarity_skipgram = skipgram_model.wv.similarity(word1, word2)
        similarity_cbow = cbow_model.wv.similarity(word1, word2)
        print(
            f"{word1} and {word2}: Skip-gram similarity = {similarity_skipgram:.4f}, CBOW similarity = {similarity_cbow:.4f}"
        )

    # Comparing syntactic relationships
    print("\nComparing Syntactic Relationships:")
    for word1, word2 in syntactic_pairs:
        similarity_skipgram = skipgram_model.wv.similarity(word1, word2)
        similarity_cbow = cbow_model.wv.similarity(word1, word2)
        print(
            f"{word1} and {word2}: Skip-gram similarity = {similarity_skipgram:.4f}, CBOW similarity = {similarity_cbow:.4f}"
        )


# Function to train FastText model (subword embeddings)
def train_fasttext_model(sentences, window_size=5):
    fasttext_model = FastText(
        sentences, vector_size=64, window=window_size, min_count=1
    )
    return fasttext_model


# Function to get the FastText embedding for a word
def get_fasttext_embedding(word, fasttext_model):
    return fasttext_model.wv[word]
