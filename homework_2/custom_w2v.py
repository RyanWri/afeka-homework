import torch
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from ollama import chat
from ollama import ChatResponse


def generate_random_sentence_ollama(model, num_sentences, file_name):
    sentences = []
    for _ in range(num_sentences):
        prompt = (
            "Pick a random topic, then generate a valid sentence with proper grammar for it. "
            "The sentence should hold 6 to 12 words with no special characters. "
            "Each topic should be totally different from the previous one. "
            "Your response should return only the sentence."
        )
        response: ChatResponse = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        sentences.append(response["message"]["content"].strip())

    with open(file_name, "a") as file:
        # Add newline after each sentence to ensure proper formatting
        file.write("\n".join(sentences) + "\n")

    print(f"{num_sentences} random sentences generated and saved to {file_name}")


generate_random_sentence_ollama(
    model="mistral", num_sentences=5, file_name="data/custom_corpus.txt"
)


# Load the trained Word2Vec model
model = Word2Vec.load("word2vec_custom.model")


# Function to get the word embedding
def get_word_embedding(word):
    try:
        return model.wv[word]
    except KeyError:
        print(f"Word '{word}' not in vocabulary.")
        return None


# Word pairs for evaluation
word_pairs = [("king", "queen"), ("man", "woman"), ("apple", "orange")]


# Function to calculate cosine similarity for word pairs
def evaluate_word_pairs(word_pairs):
    # Calculate cosine similarity for each word pair
    for word1, word2 in word_pairs:
        vec1 = get_word_embedding(word1)
        vec2 = get_word_embedding(word2)
        if vec1 is not None and vec2 is not None:
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            print(
                f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}"
            )


# Evaluate the word pairs
evaluate_word_pairs(word_pairs)
