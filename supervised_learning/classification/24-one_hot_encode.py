#!/usr/bin/env python3
"""This module includes the function one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """This function converts a numeric label vector into a one-hot matrix"""
    if max(Y) > classes:
        return None
    
    a = np.array(Y)
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b.T
