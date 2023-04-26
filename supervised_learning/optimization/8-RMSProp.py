#!/usr/bin/env python3
"""This module contains the function create_RMSProp_op"""


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """that creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm"""
    return tf.train.RMSPropOptimizer(alpha, 
                                     momentum=beta2, 
                                     epsilon=epsilon).minimize(loss)
