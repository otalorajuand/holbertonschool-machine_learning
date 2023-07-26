#!/usr/bin/env python3
"""This module contains the class GaussianProcess"""
import numpy as np


class GaussianProcess:
    """represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """init method

        Params:

        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the
        black-box function
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        kernel function aka(covariance function)
        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)
        Returns: covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """

        # formula κ(xi,xj)=σ^2f exp(−12l2(xi−xj)T(xi−xj))(6)
        # source: http://krasserm.github.io/2018/03/19/gaussian-processes/
        sqdist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """predicts the mean and standard deviation of points in a
           Gaussian process

        X_s is a numpy.ndarray of shape (s, 1) containing all of
        the points whose mean and standard deviation should be calculated

        Returns: mu, sigma
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Equation (7)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)

        # Equation (8)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s.reshape(-1), cov_s.diagonal()
