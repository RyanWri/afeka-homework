import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def train_skipgram_model(corpus, vector_size, window, min_count):
    model = Word2Vec(
        corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        alpha=0.001,
    )
    return model


def train_cbow_model(corpus, vector_size, window, min_count):
    model = Word2Vec(
        corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,
        alpha=0.001,
    )
    return model


# Function to read sentences from a file
def read_sentences(file_path):
    with open(file_path, "r") as file:
        sentences = file.readlines()
    return [word_tokenize(sentence.lower()) for sentence in sentences]


# General function to save embeddings for both Skip-gram and CBOW models
def save_embeddings(model, filename):
    word_vectors = model.wv
    word_list = list(word_vectors.index_to_key)
    embedding_matrix = np.array([word_vectors[word] for word in word_list])

    np.save(filename, embedding_matrix)
    return word_list, embedding_matrix
