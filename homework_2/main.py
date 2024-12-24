# Load the sentences from a file
from homework_2.word_2_vec import (
    read_sentences,
    train_word2vec_models,
    compare_models_for_words,
    compare_syntactic_semantic_relationships,
    train_fasttext_model,
    get_fasttext_embedding,
)
from homework_2.contextual import get_word_embedding

file_path = "data/sentences.txt"
sentences = read_sentences(file_path)

# Train Word2Vec models with a window size of 5
skipgram_model, cbow_model = train_word2vec_models(
    sentences, vector_size=64, window_size=3
)

# Compare cosine similarity between the models for specific words
words_to_compare = ["bank", "rose", "lead", "book", "file"]
similarities = compare_models_for_words(words_to_compare, skipgram_model, cbow_model)
for word, similarity in similarities.items():
    print(
        f"Cosine similarity between Skip-gram and CBOW for '{word}': {similarity:.4f}"
    )

# Analyze syntactic vs semantic relationships
compare_syntactic_semantic_relationships(skipgram_model, cbow_model)

# Train FastText model
fasttext_model = train_fasttext_model(sentences, window_size=5)

# Get FastText embeddings for a word (e.g., 'bank')
fasttext_embedding = get_fasttext_embedding("bank", fasttext_model)
print(f"FastText embedding for 'bank':\n", fasttext_embedding)
