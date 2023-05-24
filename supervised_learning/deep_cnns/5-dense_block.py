#!/usr/bin/env python3
"""This module contains the function dense_block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block as described in
       Densely Connected Convolutional Networks

    Params:
        - X: the output from the previous layer
        - nb_filters: an integer representing the number of filters in X
        - growth_rate: the growth rate for the dense block
        - layers: the number of layers in the dense block

    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively
    """

    initializer = K.initializers.he_normal()

    for lay in range(layers):

        # Batch normalization
        B = K.layers.BatchNormalization()(X)

        # activation
        relu = K.layers.ReLU()(B)

        # bottle neck convolution
        BN = K.layers.Conv2D(filters=4 * growth_rate,
                             kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=initializer)(relu)

        # Batch normalization
        N_2 = K.layers.BatchNormalization()(BN)

        # activation
        relu_2 = K.layers.ReLU()(N_2)

        # 3x3 convolution
        C = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=initializer)(relu_2)

        nb_filters += growth_rate
        X = K.layers.concatenate([X, C])

    return X, nb_filters
