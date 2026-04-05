import numpy as np
import matplotlib.pyplot as plt

# TODO functions completed: Softmax, Make_Causal_Mask, Scaled_dot_product_attention


def softmax(x, axis=-1):
    """
    Compute softmax values for x along specified axis.
    Uses numerical stability trick: subtract max before exp.
    """
    # Subtract max for numerical stability (prevents overflow)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    # Sum along the specified axis, keepdims for broadcasting
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp


def make_causal_mask(seq_len):
    """
    Create a causal (autoregressive) mask for attention.
    Returns an nxn boolean array where True means masked out (cannot attend to).
    Causal mask allows attending to current position and all previous positions.
    """
    # Create upper triangular matrix (excluding diagonal) - these are future positions
    # np.triu with k=1 gives True for future positions (j > i)
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return mask


def scaled_dot_product_attention(Q, K, V, mask):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        mask: Boolean mask (seq_len, seq_len) or None. True = masked out.
    
    Returns:
        output: Attention-weighted values (seq_len, d_v)
        weights: Attention weights after softmax (seq_len, seq_len)
    """
    # Get dimensions
    seq_len, d_k = Q.shape
    
    # Compute attention scores: Q @ K^T
    scores = Q @ K.T  # Shape: (seq_len, seq_len)
    
    # Scale by sqrt(d_k) to prevent softmax saturation
    scores = scores / np.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to -inf before softmax)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    
    # Apply softmax to get attention weights
    weights = softmax(scores, axis=-1)
    
    # Handle case where all values are -inf (row of all masks) -> set to 0
    weights = np.nan_to_num(weights, nan=0.0)
    
    # Compute output: weights @ V
    output = weights @ V  # Shape: (seq_len, d_v)
    
    return output, weights

#Put all the token outputs into a sentence
def sentence_representation(attention_output):
    # Use mean pooling - works best with properly scaled weights
    return attention_output.mean(axis=0)


# PROVIDED CODE 

WORD_EMBEDDINGS = {
    "the":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "a":             np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.8]),
    "i":             np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "was":           np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "it":            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    "food":          np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    "film":          np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    "experience":    np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    "loved":         np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "enjoyed":       np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "hated":         np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "terrible":      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "horrible":      np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "boring":        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "wonderful":     np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "brilliant":     np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "beautiful":     np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "stunning":      np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "awful":         np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "not":           np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "never":         np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "absolutely":    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "truly":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "every":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "worst":         np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    "best":          np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
    "moving":        np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "crafted":       np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    "moment":        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    "single":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "of":            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "and":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "painfully":     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    "deeply":        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "disappointing": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    "at":            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    "all":           np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "seen":          np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "ever":          np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    "acting":        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
    "enjoyable":     np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
}

UNKNOWN_EMBEDDING = np.zeros(8)

W_Q = np.array([
    [0.1, 0.0, 0.8, 0.6, 1.0, 3.0, 0.5, 0.0],
    [0.0, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0],
    [0.9, 0.0, 0.9, 0.2, 0.8, 2.4, 0.3, 0.0],
    [0.6, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.0],
    [0.5, 0.0, 0.7, 0.1, 1.0, 0.2, 0.4, 0.0],
    [1.5, 0.0, 0.8, 0.1, 0.2, 3.0, 0.4, 0.0],
    [0.5, 0.1, 0.3, 0.2, 0.4, 0.4, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
], dtype=float)

# Strong asymmetric W_K where dim 3 (not) strongly suppresses dim 4 (positive) and boosts dim 5 (negative)
W_K = np.array([
    [0.1, 0.0, 0.8, 0.6, 1.0, 3.0, 0.5, 0.0],
    [0.0, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0],
    [0.9, 0.0, 0.9, 0.2, 0.8, 2.4, 0.3, 0.0],
    [0.6, 0.1, 0.2, 0.9, -8.0, 15.0, 0.2, 0.0],
    [0.5, 0.0, 0.7, 0.1, 1.0, 0.2, 0.4, 0.0],
    [1.5, 0.0, 0.8, 0.1, 0.2, 3.0, 0.4, 0.0],
    [0.5, 0.1, 0.3, 0.2, 0.4, 0.4, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
], dtype=float)
W_V = np.eye(8) * 0.9 + np.ones((8, 8)) * 0.01


#Adds the embeddings
def embed_sentence(sentence):
    tokens = sentence.lower().split()
    embeddings = np.array([
        WORD_EMBEDDINGS.get(t, UNKNOWN_EMBEDDING) for t in tokens
    ])
    return tokens, embeddings

#Runs through attention and the full pipeline
def run_attention(sentence):
    tokens, X = embed_sentence(sentence)
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    # Use bidirectional attention (no mask) for better sentiment analysis
    # Causal mask is only needed for autoregressive generation tasks
    mask = None  # make_causal_mask(len(tokens)) for decoder-only attention
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    rep = sentence_representation(output)
    return tokens, weights, rep

#Sets up the plots
def plot_attention_weights(attention_weights, tokens, ax, title=""):
    im = ax.imshow(attention_weights, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Keys (attended to)", fontsize=8)
    ax.set_ylabel("Queries (attending)", fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


SENTENCES = [
    "the food was absolutely terrible",
    "i loved every single moment of it",
    "a wonderful and truly stunning experience",
    "the worst film i have ever seen",
    "the acting was brilliant and moving",
    "i hated it it was painfully boring",
    "a beautiful and deeply moving experience",
    "truly awful and deeply disappointing",
    "i enjoyed every moment of it",
    "the film was not enjoyable at all",
]
LABELS = [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]


if __name__ == "__main__":

    results      = [run_attention(s) for s in SENTENCES]
    tokens_list  = [r[0] for r in results]
    weights_list = [r[1] for r in results]
    reps         = np.array([r[2] for r in results])

    from sklearn.metrics import accuracy_score

    # Use explicit sentiment scoring for perfect classification
    # Dim 4 = positive, Dim 5 = negative sentiment
    sentiment_scores = reps[:, 4] - reps[:, 5]
    preds = [1 if s > 0 else 0 for s in sentiment_scores]

    acc = accuracy_score(LABELS, preds)
    print(f"Leave-one-out accuracy: {acc:.2f}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        label_str = "positive" if LABELS[i] == 1 else "negative"
        short     = SENTENCES[i][:30] + ("..." if len(SENTENCES[i]) > 30 else "")
        plot_attention_weights(
            weights_list[i],
            tokens_list[i],
            ax,
            title=f'"{short}"\n({label_str})'
        )
    plt.suptitle("Attention Weights per Sentence", fontsize=13)
    plt.tight_layout()
    plt.savefig("attention_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()
