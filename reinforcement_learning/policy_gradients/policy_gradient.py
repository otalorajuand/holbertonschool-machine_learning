#!/usr/bin/env python3
"""This module includes the function policy"""
import numpy as np


def policy(matrix, weight):
    """computes to policy with a weight of a matrix
    """

    x = matrix @ weight
    return np.exp(x) / np.exp(x).sum()
