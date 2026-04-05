import numpy as np
import matplotlib.pyplot as plt

#TODO Softmax, Make_Causal_Mask, Scaled_dot_product_attention


def softmax(x, axis=-1): #Input is a numpy matrix like [[1 2 3] [4 5 6]]. Over the axis you need to perform softmax on the values and return the resulting arrays where the values add up to 1.
    return x


def make_causal_mask(seq_len): #Input is a positive integer, 
    #Output should be a nxn array of booleans where n = seq_len. The values should be True if the position should be masked out and should be False if the position should not be masked out.
    return np.zeros((seq_len, seq_len))


def scaled_dot_product_attention(Q, K, V, mask): #Input is the Query, Key, and Value matrices and mask which if None means no masking otherwise means yes masking. 
    #There should be two outputs. #1 is output which what you would add to each embedding. #2 is the weights which is the weights AFTER softmax but before multiplying by V
    return output, weights

#Put all the token outputs into a sentence
def sentence_representation(attention_output):
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
    [0.1, 0.0, 0.8, 0.6, 0.7, 0.7, 0.5, 0.0],
    [0.0, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0],
    [0.9, 0.0, 0.9, 0.2, 0.8, 0.8, 0.3, 0.0],
    [0.6, 0.1, 0.2, 0.9, 0.1, 0.1, 0.2, 0.0],
    [0.8, 0.0, 0.7, 0.1, 0.9, 0.2, 0.4, 0.0],
    [0.7, 0.0, 0.8, 0.1, 0.2, 0.9, 0.4, 0.0],
    [0.5, 0.1, 0.3, 0.2, 0.4, 0.4, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
], dtype=float)

W_K = W_Q.copy()
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
    seq_len = len(tokens)
    mask = make_causal_mask(seq_len)
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

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score

    #Logistic Regression classifier that uses your attention weights to determine if the movie reviews are positive or negative
    loo   = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(reps):
        clf = LogisticRegression()
        clf.fit(reps[train_idx], np.array(LABELS)[train_idx])
        preds.append(clf.predict(reps[test_idx])[0])

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

