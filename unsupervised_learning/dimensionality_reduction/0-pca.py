#!/usr/bin/env python3
"""this module includes the function pca"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset

    Params:
        X is a numpy.ndarray of shape (n, d)
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var is the fraction of the variance that the PCA transformation
        should maintain

    Returns: the weights matrix, W, that maintains var fraction of X
             original variance
    """
    # U singular_v, Sigma singular_v, Vh right singular_v
    u, Sigma, vh = np.linalg.svd(X, full_matrices=False)
    cumulative_var = np.cumsum(Sigma) / np.sum(Sigma)
    r = (np.argwhere(cumulative_var >= var))[0, 0]
    w = vh.T
    wr = w[:, :r + 1]
    return wr
