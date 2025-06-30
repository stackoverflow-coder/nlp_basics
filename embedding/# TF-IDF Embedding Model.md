# TF-IDF Embedding Model

## Overview

TF-IDF (Term Frequency–Inverse Document Frequency) is a numerical statistic used to reflect how important a word is to a document in a collection or corpus. It is commonly used in information retrieval and text mining.

This repository contains a simple Python implementation of a TF-IDF embedding model. The model can build a vocabulary from a set of texts and convert new texts into TF-IDF vectors.

---

## How TF-IDF Works

TF-IDF combines two metrics:

- **Term Frequency (TF):** Measures how frequently a term appears in a document.
- **Inverse Document Frequency (IDF):** Measures how important a term is by reducing the weight of terms that appear in many documents.

The TF-IDF score for a word in a document is calculated as:

```
TF-IDF(word, document) = TF(word, document) * IDF(word)
```

Where:
- **TF(word, document):** Number of times the word appears in the document.
- **IDF(word):** `log((N + 1) / (df + 1)) + 1`
    - `N` is the total number of documents.
    - `df` is the number of documents containing the word.

---

## Usage

1. **Build the Vocabulary:**
   - The model scans all texts, selects the most common words, and computes IDF values.

2. **Convert Text to Vector:**
   - Each text is tokenized, and a vector is created where each element is the TF-IDF score of a word in the vocabulary.

---

## Applications

- Text similarity and clustering
- Information retrieval
- Feature extraction for machine learning models

---

## References

-