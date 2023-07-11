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

    maxs = np.max(X, axis=0)
    mins = np.min(X, axis=0)

    res = np.random.uniform(mins, maxs, (k, 2))
    return res
