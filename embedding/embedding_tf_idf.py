import numpy as np
import re
from collections import Counter, defaultdict

class TFIDFEmbeddingModel:
    """
    A simple TF-IDF embedding model for converting text documents into vector representations.
    Attributes:
        vocab_size (int): Maximum number of words in the vocabulary.
        word2idx (dict): Mapping from words to their corresponding indices in the vocabulary.
        idx2word (dict): Mapping from indices to their corresponding words in the vocabulary.
        idf (dict): Inverse Document Frequency (IDF) values for words in the vocabulary.
        vocab_built (bool): Indicates whether the vocabulary has been built.
    Methods:
        tokenize(text):
            Tokenizes the input text into lowercase words using regular expressions.
        build_vocab(texts):
            Builds the vocabulary from a list of texts and computes the IDF values for each word.
            Only the top `vocab_size` most frequent words are included in the vocabulary.
        text_to_vector(text):
            Converts a given text into its TF-IDF vector representation using the built vocabulary.
            Raises:
                ValueError: If the vocabulary has not been built prior to calling this method.
    """
    def __init__(self, vocab_size=1000):
        # Maximum number of words in the vocabulary
        self.vocab_size = vocab_size
        self.word2idx = {}    # Maps words to indices
        self.idx2word = {}    # Maps indices to words
        self.idf = {}         # Stores IDF values for words
        self.vocab_built = False

    def tokenize(self, text):
        # Tokenize text into lowercase words using regex
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, texts):
        # Build vocabulary and compute IDF from a list of texts
        df = defaultdict(int)  # Document frequency for each word
        all_tokens = []
        for text in texts:
            tokens = set(self.tokenize(text))  # Unique tokens in the document
            all_tokens.extend(tokens)
            for token in tokens:
                df[token] += 1
        # Select top vocab_size words by frequency
        most_common = Counter(all_tokens).most_common(self.vocab_size)
        vocab = [word for word, _ in most_common]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        # Compute IDF for each word in the vocabulary
        N = len(texts)
        self.idf = {word: np.log((N + 1) / (df[word] + 1)) + 1 for word in vocab}
        self.vocab_built = True

    def text_to_vector(self, text):
        # Convert a text to its TF-IDF vector representation
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        tokens = self.tokenize(text)
        tf = Counter(tokens)  # Term frequency in the text
        vec = np.zeros(len(self.word2idx))
        for word, idx in self.word2idx.items():
            # Multiply term frequency by IDF
            vec[idx] = tf[word] * self.idf.get(word, 0.0)
        return vec

if __name__ == "__main__":
    # Example usage
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A fox is quick and brown."
    ]
    model = TFIDFEmbeddingModel(vocab_size=10)
    model.build_vocab(texts)
    for text in texts:
        vec = model.text_to_vector(text)
        print(f"Text: {text}")
        print(f"Vector: {vec}\n")