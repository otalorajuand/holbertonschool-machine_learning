#!/usr/bin/env python3
"""This module contains the function calculate_accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    with tf.name_scope("layer"):
        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                  predictions=tf.argmax(y_pred, 1))
    return acc_op
