#!/usr/bin/env python3
"""This module contains the function one_hot"""
import numpy as np


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix

    Params:
        labels: a list with the labels
        classes: the number of classes

    Returns: the one-hot matrix
    """

    classes = len(labels) if classes is None else classes
    b = np.zeros((labels.size, classes))
    b[np.arange(labels.size), labels] = 1
    return b
