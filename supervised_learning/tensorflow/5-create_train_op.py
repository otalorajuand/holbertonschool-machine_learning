#!/usr/bin/env python3
"""This module contains the function create_train_op"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
