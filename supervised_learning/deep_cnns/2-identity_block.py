#!/usr/bin/env python3
"""this module contains the function identity_block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """builds an identity block as described in
       Deep Residual Learning for Image Recognition (2015)

    Params:
        - A_prev: the output from the previous layer
        - filters is a tuple or list containing F11, F3, F12, respectively:
            * F11 is the number of filters in the first 1x1 convolution
            * F3 is the number of filters in the 3x3 convolution
            * F12 is the number of filters in the second 1x1 convolution

    Returns: the activated output of the identity block
    """

    initializer = K.initializers.he_normal()
    F11, F3, F12 = filters

    # 1x1 convolution
    C1 = K.layers.Conv2D(filters=F11,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=initializer)(A_prev)

    # Batch normalization
    BN_1 = K.layers.BatchNormalization(axis=3)(C1)

    # activation
    relu_1 = K.layers.Activation('relu')(BN_1)

    # 3x3 convolution
    C2 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=initializer)(relu_1)

    # Batch normalization
    BN_2 = K.layers.BatchNormalization(axis=3)(C2)

    # activation
    relu_2 = K.layers.Activation('relu')(BN_2)

    # 1x1 convolution
    C3 = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=initializer)(relu_2)

    # Batch normalization
    BN_3 = K.layers.BatchNormalization(axis=3)(C3)

    # Addition
    add = K.layers.Add()([A_prev, BN_3])

    # Activation
    relu_3 = K.layers.Activation('relu')(add)

    return relu_3