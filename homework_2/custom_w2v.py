from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from ollama import chat


def generate_random_sentence_ollama(model, num_sentences, file_name):
    sentences = []
    for _ in range(num_sentences):
        prompt = """
           Pick a random topic related to royalty, gender, or fruits, and generate a valid sentence 
           with proper grammar. The sentence should include one of the following word pairs: 
           "king" and "queen", "man" and "woman", or "apple" and "orange". 
           The sentence should be between 6 to 12 words long with no special characters. 
           Ensure that each sentence is unique and the topics are different from each other.
           """
        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        sentences.append(response["message"]["content"].strip())

    with open(file_name, "w") as file:
        # Add newline after each sentence to ensure proper formatting
        file.write("\n".join(sentences) + "\n")

    print(f"{num_sentences} random sentences generated and saved to {file_name}")


# Function to get the word embedding
def get_word_embedding(model, word):
    try:
        return model.wv[word]
    except KeyError:
        print(f"Word '{word}' not in vocabulary.")
        return None


def evaluate_word_pairs(model, word_pairs):
    # Calculate cosine similarity for each word pair
    for word1, word2 in word_pairs:
        vec1 = get_word_embedding(model, word1)
        vec2 = get_word_embedding(model, word2)
        if vec1 is not None and vec2 is not None:
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            print(
                f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}"
            )


if __name__ == "__main__":
    # Generate random sentences and save them to a file
    generate_random_sentence_ollama(
        model="mistral", num_sentences=30, file_name="data/custom_corpus.txt"
    )

    # Load the trained Word2Vec model (ensure the model has been trained and saved)
    model = Word2Vec.load("word2vec_custom.model")

    # Word pairs for evaluation
    word_pairs = [("king", "queen"), ("man", "woman"), ("apple", "orange")]

    # Evaluate the word pairs
    evaluate_word_pairs(model, word_pairs)
