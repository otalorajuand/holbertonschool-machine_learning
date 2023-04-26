#!/usr/bin/env python3
"""This module contains the function shuffle_data"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    m = np.random.permutation(X.shape[0])
    X_perm = X[m]
    Y_perm = Y[m]
    return X_perm, Y_perm
