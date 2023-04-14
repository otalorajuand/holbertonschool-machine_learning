#!/usr/bin/env python3
"""This module includes the function one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """This function converts a numeric label vector into a one-hot matrix"""
    if max(Y) > classes or not isinstance(Y, np.ndarray):
        return None

    b = np.zeros((Y.size, Y.max() + 1))
    b[np.arange(Y.size), Y] = 1
    return b.T
