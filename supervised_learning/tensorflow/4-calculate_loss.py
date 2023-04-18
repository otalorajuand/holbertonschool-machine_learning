#!/usr/bin/env python3
"""This module contains the function calculate_loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    y_pred = tf.math.argmax(y_pred, axis=1)
    y = tf.math.argmax(y, axis=1)
    return tf.losses.softmax_cross_entropy(y, y_pred)
