#!/usr/bin/env python3
"""This module contains the function pca"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset

    Params:
        X is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
        ndim is the new dimensionality of the transformed X

    Returns: T, a numpy.ndarray of shape (n, ndim) containing
             the transformed version of X
    """

    # U singular_v, Sigma singular_v, Vh right singular_v
    X_mean = X - np.mean(X, axis=0)
    u, Sigma, vh = np.linalg.svd(X_mean)
    w = vh.T
    wr = w[:, :ndim]

    return np.matmul(X_mean, wr)
