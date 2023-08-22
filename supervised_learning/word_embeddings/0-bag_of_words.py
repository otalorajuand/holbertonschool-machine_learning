#!/usr/bin/env python3
"""This module contains the function bag_of_words"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """creates a bag of words embedding matrix

    Args:
    - sentences is a list of sentences to analyze
    - vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used

    Return: embeddings, features
    - embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
        s is the number of sentences in sentences
        f is the number of features analyzed
    - features is a list of the features used for embeddings
    """
    if not vocab:
        vocab = []

        for sentence in sentences:
            # Convert to lowercase and split by spaces
            words = sentence.lower().split()

            # Remove non-letter symbols using regular expression
            words = [re.sub(r'[^a-z]', '', word) for word in words]

            # Remove empty strings resulting from symbols removal
            words = [word for word in words if word]

            vocab.extend(words)

        vocab = sorted(list(set(vocab)))

    n_sentences = len(sentences)
    n_vocab = len(vocab)
    embeddings = np.zeros((n_sentences, n_vocab))

    for i, row in enumerate(embeddings):
        for j, elem in enumerate(row):
            if vocab[j] in sentences[i].lower():
                embeddings[i, j] = 1

    return embeddings, vocab
