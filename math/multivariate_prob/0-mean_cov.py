#!/usr/bin/env python3
"""This module contains the function mean_cov"""
import numpy as np


def mean_cov(X):
    """calculates the mean and covariance of a data set

    Params:
        X: a numpy.ndarray of shape (n, d) containing the data set:
            n: the number of data points
            d: the number of dimensions in each data point

    Returns: mean, cov:
        mean: a numpy.ndarray of shape (1, d) containing
              the mean of the data set
        cov: a numpy.ndarray of shape (d, d) containing
             the covariance matrix of the data set
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n = X.shape[0]

    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.sum(X, axis=0) / n

    Z = (X - mean)
    cov = np.matmul(Z.T, Z) / n

    return mean, cov
