#!/usr/bin/env python3
"""This module contains the function densenet121"""
import tensorflow.keras as K


dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """builds the DenseNet-121 architecture as described
       in Densely Connected Convolutional Networks

    Params:
        - growth_rate: the growth rate
        - compression: the compression factor

    Returns: the keras model
    """

    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()

    # Batch normalization
    BN_1 = K.layers.BatchNormalization()(X)

    # activation
    relu_1 = K.layers.ReLU()(BN_1)

    # 7x7 convolution
    C1 = K.layers.Conv2D(filters=2 * growth_rate,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=2,
                         kernel_initializer=initializer)(relu_1)

    # maxpool
    MP_2 = K.layers.MaxPool2D(pool_size=(3, 3),
                              padding='same',
                              strides=2)(relu_1)

    dense1, np1 = dense_block(MP_2, 2 * growth_rate, growth_rate, 6)
    trans1, np2 = transition_layer(dense1, np1, compression)

    dense2, np3 = dense_block(trans1, np2, growth_rate, 12)
    trans2, np4 = transition_layer(dense2, np3, compression)

    dense3, np5 = dense_block(trans2, np4, growth_rate, 24)
    trans3, np6 = transition_layer(dense2, np5, compression)

    dense4, np7 = dense_block(trans3, np6, growth_rate, 16)

    AP = K.layers.AveragePooling2D(pool_size=(7, 7),
                                   padding='valid')(dense4)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(AP)

    model = K.models.Model(inputs=X, outputs=linear)

    return model
