#!/usr/bin/env python3
"""This module contains the function gensim_to_keras"""
from gensim.models import KeyedVectors


def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer

    Args:
    - model is a trained gensim word2vec models

    Returns: the trainable keras Embedding
    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
    )
    return layer
