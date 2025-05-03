import torch
from transformers import BertTokenizer, BertModel


# Function to get contextual embeddings for a word using BERT
def get_contextual_embedding(sentence, word, model, tokenizer):
    # Tokenize the input sentence and get token IDs
    inputs = tokenizer(sentence, return_tensors="pt")

    # Get the embeddings from BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Tokenized sentence to find word's token indices
    tokenized_sentence = tokenizer.tokenize(sentence)
    word_tokens = tokenizer.tokenize(word)

    # Find the start and end indices of the word tokens in the sentence
    start_idx = tokenized_sentence.index(word_tokens[0])
    end_idx = start_idx + len(word_tokens) - 1

    # Get the embeddings for the word's tokens
    word_embedding = outputs.last_hidden_state[0, start_idx : end_idx + 1, :]

    # Average the embeddings for the word's tokens (since words can be split into subwords)
    word_embedding_avg = word_embedding.mean(dim=0).numpy()
    return word_embedding_avg


def get_word_embedding(sentence1, sentence2):
    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Get embeddings for "lead" in both contexts
    embedding1 = get_contextual_embedding(sentence1, "lead", model, tokenizer)
    embedding2 = get_contextual_embedding(sentence2, "lead", model, tokenizer)

    return embedding1, embedding2


def compute_cosine_similarity(vec1, vec2):
    # Compute cosine similarity between two vectors
    cos_sim = torch.nn.functional.cosine_similarity(
        torch.tensor(vec1), torch.tensor(vec2), dim=0
    )
    return cos_sim.item()
