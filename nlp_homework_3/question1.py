import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_weights(attention_weights, tokens, title):
    """
    Visualize attention weights as a heatmap.

    Parameters:
    - attention_weights (np.ndarray): Attention weights (seq_len, seq_len).
    - tokens (list of str): Tokens in the sentence.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True,
    )
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()


def self_attention(queries, keys, values, mask=None):
    """
    Implement vanilla self-attention.

    Parameters:
    - queries (np.ndarray): Query matrix of shape (batch_size, seq_len, d_k).
    - keys (np.ndarray): Key matrix of shape (batch_size, seq_len, d_k).
    - values (np.ndarray): Value matrix of shape (batch_size, seq_len, d_v).
    - mask (np.ndarray or None): Optional mask matrix of shape (batch_size, seq_len, seq_len).

    Returns:
    - np.ndarray: Attention output of shape (batch_size, seq_len, d_v).
    """
    d_k = queries.shape[-1]  # Dimensionality of keys/queries

    # Step 1: Compute attention scores
    scores = np.matmul(
        queries, keys.transpose(0, 2, 1)
    )  # (batch_size, seq_len, seq_len)
    scores /= np.sqrt(d_k)  # Scale scores by sqrt(d_k)

    # Step 2: Apply mask (if provided)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 3: Compute attention weights
    attention_weights = np.exp(scores)
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)

    # Step 4: Compute the attention output
    attention_output = np.matmul(
        attention_weights, values
    )  # (batch_size, seq_len, d_v)

    return attention_output, attention_weights


def multi_head_attention(x, num_heads=8):
    """
    Implement multi-head attention with head splitting.

    Parameters:
    - x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model).
    - num_heads (int): Number of attention heads.

    Returns:
    - np.ndarray: Output tensor of shape (batch_size, seq_len, d_model).
    """
    batch_size, seq_len, d_model = x.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    d_k = d_model // num_heads  # Dimensionality per head

    # Step 1: Linear projection for queries, keys, and values
    queries = x @ np.random.randn(d_model, d_model)
    keys = x @ np.random.randn(d_model, d_model)
    values = x @ np.random.randn(d_model, d_model)

    # Step 2: Split into multiple heads
    queries = queries.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    keys = keys.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Step 3: Apply self-attention to each head
    attention_weights_all_heads = []
    attention_outputs = []

    for h in range(num_heads):
        attention_output, attention_weights = self_attention(
            queries[:, h], keys[:, h], values[:, h]
        )
        attention_outputs.append(attention_output)
        attention_weights_all_heads.append(attention_weights)

    # Step 4: Concatenate the heads
    concatenated = np.concatenate(
        attention_outputs, axis=-1
    )  # (batch_size, seq_len, d_model)

    # Concatenate weights for return
    return concatenated, np.array(attention_weights_all_heads)


if __name__ == "__main__":
    # Define polysemy examples
    sentence1 = "The lead actor delivered an incredible performance."
    sentence2 = "Exposure to lead is harmful to health."

    # Mock attention weights for visualization (replace with actual weights)
    tokens_sentence1 = sentence1.split()
    tokens_sentence2 = sentence2.split()
    attention_weights_mock = np.random.rand(
        len(tokens_sentence1), len(tokens_sentence1)
    )

    plot_attention_weights(
        attention_weights_mock, tokens_sentence1, "Self-Attention Weights: Sentence 1"
    )

    # Mock embeddings (batch_size=1 for simplicity)
    # For simplicity, each word is represented as a random vector of size 8 (d_model=8).
    np.random.seed(42)  # Ensure reproducibility
    embeddings_sentence1 = np.random.rand(1, len(sentence1.split()), 8)
    embeddings_sentence2 = np.random.rand(1, len(sentence2.split()), 8)

    # Apply self_attention
    attention_output_sentence1, weights_sentence1 = self_attention(
        embeddings_sentence1, embeddings_sentence1, embeddings_sentence1
    )
    attention_output_sentence2, weights_sentence2 = self_attention(
        embeddings_sentence2, embeddings_sentence2, embeddings_sentence2
    )

    # Plot actual weights
    plot_attention_weights(
        weights_sentence1[0], tokens_sentence1, "Self-Attention Weights: Sentence 1"
    )
    plot_attention_weights(
        weights_sentence2[0], tokens_sentence2, "Self-Attention Weights: Sentence 2"
    )

    # Apply multi_head_attention
    multi_head_output_sentence1 = multi_head_attention(embeddings_sentence1)
    multi_head_output_sentence2 = multi_head_attention(embeddings_sentence2)

    # Print results
    print("Self-Attention Output for Sentence 1:\n", attention_output_sentence1)
    print("\nSelf-Attention Output for Sentence 2:\n", attention_output_sentence2)

    print(
        "\nMulti-Head Attention Output for Sentence 1:\n", multi_head_output_sentence1
    )
    print(
        "\nMulti-Head Attention Output for Sentence 2:\n", multi_head_output_sentence2
    )
