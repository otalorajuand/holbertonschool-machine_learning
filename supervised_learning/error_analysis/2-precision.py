#!/usr/bin/env python3
"""this module contains the function precision"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix

    Params:
        confusion: confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels

    Returns: numpy.ndarray of shape (classes,) containing
    the precision of each class
    """

    confusion_diag = np.diag(confusion)
    confusion_cols_sum = np.sum(confusion, axis=0)

    precision = confusion_diag / confusion_cols_sum
    return precision
