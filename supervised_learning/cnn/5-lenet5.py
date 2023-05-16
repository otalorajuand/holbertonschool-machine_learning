#!/usr/bin/env python3
"""This module contains lenet5"""
import tensorflow.keras as K


def lenet5(X):
    """builds a modified version of the LeNet-5 architecture using keras

    Params:
        - X is a K.Input of shape (m, 28, 28, 1) containing
          the input images for the network

    Returns: a K.Model compiled to use Adam optimization
             (with default hyperparameters) and accuracy metrics
    """

    initializer = K.initializers.he_normal()

    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_1 = C1(X)

    P2 = K.layers.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))
    output_2 = P2(output_1)

    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='valid',
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_3 = C3(output_2)

    P4 = K.layers.MaxPool2D(pool_size=(2, 2),
                            strides=(2, 2))
    output_4 = P4(output_3)

    output_5 = K.layers.Flatten()(output_4)

    FC6 = K.layers.Dense(units=120,
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_6 = FC6(output_5)

    FC7 = K.layers.Dense(units=84,
                         kernel_initializer=initializer,
                         activation=K.activations.relu)
    output_7 = FC7(output_6)

    FC8 = K.layers.Dense(units=10,
                         kernel_initializer=initializer)
    output_8 = FC8(output_7)

    softmax = K.layers.Softmax()(output_8)

    model = K.Model(inputs=X, outputs=softmax)

    optimizer = K.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
