#!/usr/bin/env python3
"""This module includes the function inception_network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """builds the inception network as described in Going Deeper
       with Convolutions (2014)

    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()

    C1 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=(2, 2),
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_1 = C1(X)

    MP_1 = K.layers.MaxPool2D(pool_size=(3, 3),
                              padding='same',
                              strides=(2, 2))
    output_MP_1 = MP_1(output_1)

    C2 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         padding='same',
                         strides=(1, 1),
                         activation=K.activations.relu,
                         kernel_initializer=initializer)(output_MP_1)

    C3 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_2 = C3(C2)

    MP_2 = K.layers.MaxPool2D(pool_size=(3, 3),
                              padding='same',
                              strides=(2, 2))
    output_MP_2 = MP_2(output_2)

    INC_3a = inception_block(output_MP_2, [64, 96, 128, 16, 32, 32])
    INC_3b = inception_block(INC_3a, [128, 128, 192, 32, 96, 64])

    MP_3 = K.layers.MaxPool2D(pool_size=(3, 3),
                              padding='same',
                              strides=(2, 2))
    output_MP_3 = MP_3(INC_3b)

    INC_4a = inception_block(output_MP_3, [192, 96, 208, 16, 48, 64])
    INC_4b = inception_block(INC_4a, [160, 112, 224, 24, 64, 64])
    INC_4c = inception_block(INC_4b, [128, 128, 256, 24, 64, 64])
    INC_4d = inception_block(INC_4c, [112, 144, 288, 32, 64, 64])
    INC_4e = inception_block(INC_4d, [256, 160, 320, 32, 128, 128])

    MP_4 = K.layers.MaxPool2D(pool_size=(3, 3),
                              padding='same',
                              strides=(2, 2))
    output_MP_4 = MP_4(INC_4e)

    INC_5a = inception_block(output_MP_4, [256, 160, 320, 32, 128, 128])
    INC_5b = inception_block(INC_5a, [384, 192, 384, 48, 128, 128])

    AP = K.layers.AveragePooling2D(pool_size=(7, 7),
                                   strides=(1, 1),
                                   padding='valid')
    output_AP = AP(INC_5b)

    dropout = K.layers.Dropout(0.4)(output_AP)

    linear = K.layers.Dense(units=1000,
                            activation='softmax',
                            kernel_initializer=initializer)(dropout)

    model = K.models.Model(inputs=X, outputs=linear)

    return model
