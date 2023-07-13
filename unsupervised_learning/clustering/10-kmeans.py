#!/usr/bin/env python3
"""This module containst the function kmeans"""
import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset

    Params:
        X: a numpy.ndarray of shape (n, d) containing the dataset
        k: the number of clusters

    Returns: C, clss
        C: a numpy.ndarray of shape (k, d) containing the centroid
           means for each cluster
        clss: a numpy.ndarray of shape (n,) containing the index
              of the cluster in C that each data point belongs to
    """

    kmeans = sklearn.cluster.KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)

    return kmeans.cluster_centers_, kmeams.labels_
