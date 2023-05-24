#!/usr/bin/env python3
"""This module contains the function transition_layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """builds a transition layer as described in
       Densely Connected Convolutional Networks

    Params:
        - X: the output from the previous layer
        - nb_filters: an integer representing the number of filters in X
        - compression: the compression factor for the transition layer

    Returns: The output of the transition layer
             and the number of filters within the output, respectively
    """

    # Batch normalization
    B = K.layers.BatchNormalization()(X)

    # activation
    relu = K.layers.ReLU()(B)

    # 3x3 convolution
    C = K.layers.Conv2D(filters=int(nb_filters * compression),
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=initializer)(relu)

    # Average Pooling
    AP = K.layers.AveragePooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(C)

    return AP, int(nb_filters * compression)
