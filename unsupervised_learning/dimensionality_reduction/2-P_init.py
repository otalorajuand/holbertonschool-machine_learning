#!/usr/bin/env python3
"""Initialize SNE"""

import numpy as np


def P_init(X, perplexity):
    """
    Initialize variables to calculate affinities t-SNE
    Args:
        X: numpy.ndarray of shape (n, d) containing the
           dataset to be transformed by t-SNE
        perplexity: perplexity that all Gaussian distributions should have
    Returns: (D, P, betas, H)
            D: a numpy.ndarray of shape (n, n) that calculates the pairwise
               distance between two data points
            P: a numpy.ndarray of shape (n, n) initialized to all 0s that
               will contain the P affinities
            betas: a numpy.ndarray of shape (n, 1) initialized to all 1s
            that will contain all of the beta values
            H: is the Shannon entropy for perplexity perplexity
    """
    n, _ = X.shape
    # e distance for all pairs of points in input matrix X
    # ||Xi - Xj|| ** 2
    sum_X = np.sum(np.square(X), axis=1)
    # exp(-||Yi - Yj|| ** 2)
    D = (np.add(np.add(-2 * np.matmul(X, X.T), sum_X).T, sum_X))
    # fill diagonal of 0
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
