#!/usr/bin/env python3
"""this module contains the function f1_score"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix

    Params:
        confusion: a confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and column indices represent
        the predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing
    the F1 score of each class
    """

    precision_value = precision(confusion)
    sensitivity_value = sensitivity(confusion)

    f1 = (2 * precision_value * sensitivity_value) / \
        (precision_value + sensitivity_value)

    return f1
