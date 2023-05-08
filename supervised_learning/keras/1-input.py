#!/usr/bin/env python3
"""This module contains the function build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library

    Params:
        nx: the number of input features to the network
        layers: a list containing the number of nodes in each layer
                of the network
        activations: a list containing the activation functions used
                     for each layer of the network
        lambtha: the L2 regularization parameter
        keep_prob: the probability that a node will be kept for dropout

    Returns: the keras model
    """
    L2 = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    outputs = K.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_regularizer=L2,
                             name='dense')(inputs)

    for i in range(1, len(layers)):
        dropout = K.layers.Dropout(1 - keep_prob)(outputs)
        outputs = K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=L2,
                                 name='dense_' + str(i))(dropout)

    return K.Model(inputs=inputs, outputs=outputs)
