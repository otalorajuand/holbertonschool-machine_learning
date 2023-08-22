#!/usr/bin/env python3
"""This module contains the function tf_idf"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding

    Args:
    - sentences is a list of sentences to analyze
    - vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used

    Returns: embeddings, features
    - embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
        s is the number of sentences in sentences
        f is the number of features analyzed
    - features is a list of the features used for embeddings
    """
    if vocab is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)

    embedding = X.toarray()

    return embedding, vocab
