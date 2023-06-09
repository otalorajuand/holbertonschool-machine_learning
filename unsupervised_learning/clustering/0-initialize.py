#!/usr/bin/env python3
"""This module contains the function initialize"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means

    Params:
        X: a numpy.ndarray of shape (n, d) containing the dataset that will
           be used for K-means clustering
        k: a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
             centroids for each cluster, or None on failure
    """

    if not isinstance(k, int) or k <= 0:
        return None

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    n, d = X.shape

    maxs = np.max(X, axis=0)
    mins = np.min(X, axis=0)

    res = np.random.uniform(mins, maxs, (k, d))
    return res
