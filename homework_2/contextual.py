import torch
from transformers import BertTokenizer, BertModel


# Function to get contextual embeddings for a word using BERT
def get_contextual_embedding(sentence, word, model, tokenizer):
    # Tokenize the input sentence and get token IDs
    inputs = tokenizer(sentence, return_tensors="pt")

    # Get the embeddings from BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the word token index
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


def get_word_embedding():
    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Example sentences
    sentence1 = "I went to the bank to withdraw money."
    sentence2 = "The bank of the river was covered in grass."

    # Get embeddings for "bank" in both contexts
    embedding1 = get_contextual_embedding(sentence1, "bank", model, tokenizer)
    embedding2 = get_contextual_embedding(sentence2, "bank", model, tokenizer)

    # Display embeddings
    print("Contextual embedding for 'bank' in financial context:", embedding1)
    print("Contextual embedding for 'bank' in river context:", embedding2)

    return embedding1, embedding2
