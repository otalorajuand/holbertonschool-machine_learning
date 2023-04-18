#!/usr/bin/env python3
"""This module contains the function create_layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """this function creates the nn layer"""
    with tf.name_scope("layer"):
        if activation == tf.nn.tanh:
            name = 'tanh'
        else if activation == tf.nn.relu:
            name = 'relu'
        else if activation == tf.nn.sigmoid:
            name = 'sigmoid'
        else:
            name = ''

        layer = tf.layers.dense(
            prev,
            activation=activation,
            units=n,
            name=name,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"))

    return layer
