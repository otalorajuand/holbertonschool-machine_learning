#!/usr/bin/env python3
"""This module contains the function calculate_accuracy"""


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    with tf.name_scope("layer"):
        acc, _= tf.metrics.accuracy(labels=tf.argmax(y, 1), 
                                  predictions=tf.argmax(y_pred,1))

        return acc
