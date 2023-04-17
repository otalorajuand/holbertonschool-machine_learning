#!/usr/bin/env python3
"""This module includes the function create_placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """This function eturns two placeholders, x and y, for the neural network"""
    x = tf.placeholder("float", [None, nx])
    y = tf.placeholder("float", [None, classes])

    return x, y
