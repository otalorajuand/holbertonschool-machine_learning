#!/usr/bin/env python3
"""This module contains the function one_hot_decode"""
import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""

    return np.argmax(one_hot, axis=0)
