#!/usr/bin/env python3
"""This module contains the function positional_encoding"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """calculates the positional encoding for a transformer

    Args:
    - max_seq_len is an integer representing the maximum sequence length
    - dm is the model depth

    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
             positional encoding vectors
    """
    n = 1000
    P = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in np.arange(int(dm / 2)):
            denominator = np.power(n, 2 * i / dm)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P
