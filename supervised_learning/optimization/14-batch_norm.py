#!/usr/bin/env python3
"""This module contains the function create_batch_norm_layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network
    in tensorflow"""
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=weights_initialiazer)
    x = layer[prev]
    gamma = tf.Variable(tf.constant(
        1, shape=(1, n), trainable=True, name="gamma"))
    beta = tf.Variable(tf.constant(
        0, shape=(1, n), trainable=True, name="gamma"))
    Z = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    return Z
