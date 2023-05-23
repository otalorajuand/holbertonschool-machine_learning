#!/usr/bin/env python3
"""This module contains the function inception_block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ builds an inception block as described
        in Going Deeper with Convolutions (2014):

    Params:
        - A_prev: the output from the previous layer
        - filters: a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
          respectively:
            - F1 is the number of filters in the 1x1 convolution
            - F3R is the number of filters in the 1x1 convolution before
              the 3x3 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F5R is the number of filters in the 1x1 convolution
              before the 5x5 convolution
            - F5 is the number of filters in the 5x5 convolution
            - FPP is the number of filters in the 1x1 convolution
              after the max pooling

    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    initializer = K.initializers.he_normal()

    # 1x1 convolution
    C1 = K.layers.Conv2D(filters=F1,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_1 = C1(A_prev)

    # 1x1 convolution before the 3x3 convolution
    C3R = K.layers.Conv2D(filters=F3R,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=initializer,
                          activation=K.activations.relu)
    output_3R = C3R(A_prev)

    # the 3x3 convolution
    C3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_3 = C3(output_3R)

    # 1x1 convolution before the 5x5 convolution
    C5R = K.layers.Conv2D(filters=F5R,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=initializer,
                          activation=K.activations.relu)
    output_5R = C5R(A_prev)

    # the 5x5 convolution
    C5 = K.layers.Conv2D(filters=F5,
                         kernel_size=(5, 5),
                         padding='same',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_5 = C5(output_5R)

    # 3x3 max pooling
    MP = K.layers.MaxPool2D(pool_size=(3, 3),
                            padding='same',
                            strides=(1, 1))
    output_MP = MP(A_prev)

    # the 1x1 convolution after the max pooling
    CPP = K.layers.Conv2D(filters=FPP,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=initializer,
                          activation=K.activations.relu)
    output_PP = CPP(output_MP)

    filter_concat = K.layers.Concatenate(axis=-1)([output_1,
                                                   output_3,
                                                   output_5,
                                                   output_PP])

    return filter_concat
