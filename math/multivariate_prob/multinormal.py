#!/usr/bin/env python3
"""This module contains the class MultiNormal"""
import numpy as np


class MultiNormal:
    """represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        data: a numpy.ndarray of shape (d, n) containing the data set:
            n: the number of data points
            d: the number of dimensions in each data point

        Set the public instance variables:
            mean - a numpy.ndarray of shape (d, 1) containing the mean of data
            cov - a numpy.ndarray of shape (d, d) containing the covariance
                  matrix data
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        n = data.shape[1]

        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean, self.cov = self.mean_cov(data.T)

    @staticmethod
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
        cov = np.matmul(Z.T, Z) / (n - 1)

        return mean.reshape(-1, 1), cov
