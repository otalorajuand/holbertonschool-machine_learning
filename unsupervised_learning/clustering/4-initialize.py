#!/usr/bin/env python3
"""This module contains the function initialize"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables per a Gaussian Mixture Model

    Params:
        X: a numpy.ndarray of shape (n, d) containing the data set
        k: a positive integer containing the number of clusters

    Returns: pi, m, S, or None, None, None on failure
        pi: a numpy.ndarray of shape (k,) containing the priors per
            each cluster, initialized evenly
        m: a numpy.ndarray of shape (k, d) containing the centroid means
        per each cluster, initialized with K-means
        S: a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices per each cluster, initialized as identity matrices
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 1:
        return None, None, None

    n, d = X.shape

    # priors per each cluster, initialized evenly
    phi = np.ones(k) / 

    # centroid means per each cluster, initialized with K-means
    m, _ = kmeans(X, k)

    # covariance matrices per each cluster, initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    return phi, m, S
