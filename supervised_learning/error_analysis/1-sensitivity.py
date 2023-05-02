#!/usr/bin/env python3
"""this module contains the function sensitivity"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix

    Params:
        confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
    of each class
    """

    confusion_diag = np.diag(confusion)
    confusion_rows_sum = np.sum(confusion, axis=1)

    sensitivity = confusion_diag / confusion_rows_sum
    return sensitivity
