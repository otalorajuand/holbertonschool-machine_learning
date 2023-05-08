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
    K.regularizers.L2(l2=lambtha)
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer='l2'))
    model.add(K.layers.Dropout(keep_prob, input_shape=(nx,)))

    for lay, act in zip(layers[1:-1], activations[1:-1]):
        model.add(K.layers.Dense(lay, activation=act,
                                 kernel_regularizer='l2'))
        model.add(K.layers.Dropout(keep_prob))

    model.add(K.layers.Dense(layers[-1], activation=activations[-1],
                             kernel_regularizer='l2'))

    return model
