#!/usr/bin/env python3
"""This module includes the function gmm"""
import sklearn.mixture


def gmm(X, k):
    """calculates a GMM from a dataset

    Params:
        X: a numpy.ndarray of shape (n, d) containing the dataset
        k: the number of clusters

    Returns: pi, m, S, clss, bic
        pi: a numpy.ndarray of shape (k,) containing the cluster priors
        m: a numpy.ndarray of shape (k, d) containing the centroid means
        S: a numpy.ndarray of shape (k, d, d) containing the covariance
           matrices
        clss: a numpy.ndarray of shape (n,) containing the cluster
              indices for each data point
        bic: a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
             value for each cluster size tested
    """

    gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    bic = gm.bic(X)
    m = gm.means_
    pi = gm.weights_
    clss = gm.predict(X)
    S = gm.covariances_

    return pi, m, S, clss, bic
