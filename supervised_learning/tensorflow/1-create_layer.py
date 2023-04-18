#!/usr/bin/env python3
"""This module contains the function create_layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """this function creates the nn layer"""
    with tf.name_scope("layer"):
        layer = tf.layers.dense(
            prev,
            activation=activation,
            units=n,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"))

    return layer
