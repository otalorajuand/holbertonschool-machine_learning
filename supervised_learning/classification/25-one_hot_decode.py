#!/usr/bin/env python3
"""This module contains the function one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None
    
    res = np.argmax(one_hot, axis=0)

    if res is int:
        return None

    return res
