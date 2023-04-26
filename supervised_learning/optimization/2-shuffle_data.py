#!/usr/bin/env python3
"""This module contains the function shuffle_data"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    X_perm = np.random.permutation(X)
    Y_perm = np.random.permutation(Y)
    return X_perm, Y_perm
