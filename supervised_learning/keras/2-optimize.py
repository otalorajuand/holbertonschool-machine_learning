#!/usr/bin/env python3
"""This module contains the function optimize_model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimization for a keras model with
       categorical crossentropy loss and accuracy metrics.

    Params:
        network: the model to optimize
        alpha: the learning rate
        beta1: the first Adam optimization parameter 
        beta2: the second Adam optimization parameter

    Returns: None
    """
    optimizer = K.optimizers.Adam(lr = alpha, beta_1 = beta1, beta_2 = beta2)
    loss = K.losses.CategoricalCrossentropy()
    network.compile(optimizer = optimizer, loss= loss, metrics=['accuracy'])