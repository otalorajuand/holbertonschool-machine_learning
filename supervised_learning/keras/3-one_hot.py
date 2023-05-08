#!/usr/bin/env python3
"""This module contains the function one_hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix

    Params:
        labels: a list with the labels
        classes: the number of classes

    Returns: the one-hot matrix
    """

    return K.utils.to_categorical(labels, num_classes=classes)