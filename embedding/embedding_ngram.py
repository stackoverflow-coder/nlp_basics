import numpy as np
from collections import Counter

def generate_ngrams(text, n):
    tokens = text.split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def ngram_embedding_numpy(texts, ngram_range=(1, 2)):
    # Build vocabulary
    vocab = set()
    for text in texts:
        for n in range(ngram_range[0], ngram_range[1]+1):
            vocab.update(generate_ngrams(text, n))
    vocab = sorted(vocab)
    vocab_index = {ngram: idx for idx, ngram in enumerate(vocab)}

    # Create embeddings
    embeddings = np.zeros((len(texts), len(vocab)), dtype=int)
    for i, text in enumerate(texts):
        ngram_counts = Counter()
        for n in range(ngram_range[0], ngram_range[1]+1):
            ngram_counts.update(generate_ngrams(text, n))
        for ngram, count in ngram_counts.items():
            embeddings[i, vocab_index[ngram]] = count

    return embeddings, vocab

# Example usage
if __name__ == "__main__":
    texts = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "the quick dog"
    ]
    embeddings, feature_names = ngram_embedding_numpy(texts, ngram_range=(1, 2))
    print("Feature names:", feature_names)
    print("Embeddings:\n", embeddings)