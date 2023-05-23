#!/usr/bin/env python3
"""This module contains the function projection_block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block as described in
       Deep Residual Learning for Image Recognition (2015)

    Params:
        - A_prev: the output from the previous layer
        - filters: a tuple or list containing F11, F3, F12, respectively:
            * F11 is the number of filters in the first 1x1 convolution
            * F3 is the number of filters in the 3x3 convolution
            * F12 is the number of filters in the second 1x1 convolution
        - s: the stride of the first convolution in both the main path
             and the shortcut connection

    Returns: the activated output of the projection block
    """

    initializer = K.initializers.he_normal()
    F11, F3, F12 = filters

    # 1x1 convolution
    C1 = K.layers.Conv2D(filters=F11,
                         kernel_size=(1, 1),
                         padding='same',
                         strides=s,
                         kernel_initializer=initializer)(A_prev)

    # Batch normalization
    BN_1 = K.layers.BatchNormalization(axis=3)(C1)

    # activation
    relu_1 = K.layers.ReLU()(BN_1)

    # 3x3 convolution
    C2 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=initializer)(relu_1)

    # Batch normalization
    BN_2 = K.layers.BatchNormalization(axis=3)(C2)

    # activation
    relu_2 = K.layers.ReLU()(BN_2)

    # 1x1 convolution
    C3 = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=initializer)(relu_2)

    # 1x1 convolution
    C1_projection = K.layers.Conv2D(filters=F12,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    strides=s,
                                    kernel_initializer=initializer)(A_prev)

    # Batch normalization
    BN_3 = K.layers.BatchNormalization(axis=3)(C3)
    BN_3_projection = K.layers.BatchNormalization(axis=3)(C1_projection)

    # Addition
    add = K.layers.Add()([BN_3, BN_3_projection])

    # Activation
    relu_3 = K.layers.ReLU()(add)

    return relu_3
