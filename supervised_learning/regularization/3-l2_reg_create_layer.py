#!/usr/bin/env python3
"""this module contains the function l2_reg_create_layer"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization

    Params:
        prev: a tensor containing the output of the previous layer
        n: the number of nodes the new layer should contain
        activation: the activation function that should be used on the layer
        lambtha: the L2 regularization parameter

    Returns: the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    regularizer = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer)

    return layer(prev)
