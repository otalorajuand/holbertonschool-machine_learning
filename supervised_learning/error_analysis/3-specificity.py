#!/usr/bin/env python3
"""this module contains the function specificity"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix

    Params:
        confusion: a confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and column indices represent
        the predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing
    the specificity of each class
    """

    specificity_list = []
    for m in range(confusion.shape[0]):

        rows_to_sum = np.arange(confusion.shape[0]) != m
        cols_to_sum = np.arange(confusion.shape[1]) != m

        true_negatives = np.sum(confusion[rows_to_sum][:, cols_to_sum])
        false_positives = np.sum(confusion[:, m]) - confusion[m, m]

        specificity = true_negatives / (true_negatives + false_positives)
        specificity_list.append(specificity)

    return np.array(specificity_list)
