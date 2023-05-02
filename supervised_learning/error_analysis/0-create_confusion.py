#!/usr/bin/env python3
"""this module contains the function create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix

    Params:
        labels: one-hot numpy.ndarray of shape (m, classes)
                containing the correct labels
        logits: one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels

    Returns: a confusion numpy.ndarray of shape (classes, classes)
    with row indices representing the correct labels and column indices
    representing the predicted labels"""

    res_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    m = labels.shape[0]

    for i in range(m):

        col = np.where(logits[i] == 1)[0][0]
        row = np.where(labels[i] == 1)[0][0]

        res_matrix[row, col] += 1

    return res_matrix
