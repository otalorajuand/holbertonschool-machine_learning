#!/usr/bin/env python3
"""conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout

    Params:
        X: a numpy.ndarray of shape (nx, m) containing the input data
        for the network
        weights: a dictionary of the weights and biases of the neural network
        L: the number of layers in the network
        keep_prob: the probability that a node will be kept

    Returns: a dictionary containing the outputs of each layer and
    the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(L):
        W = weights['W' + str(i + 1)]
        A = cache['A' + str(i)]
        B = weights['b' + str(i + 1)]

        Z = np.dot(W, A) + B

        D = np.where(np.random.rand(Z.shape[0], Z.shape[1]) < keep_prob, 1, 0)

        if i == L - 1:
            soft_max = np.exp(Z)
            cache['A' + str(i + 1)] = (soft_max / np.sum(soft_max, axis=0,
                                                         keepdims=True))
        else:
            tanh = np.tanh(Z)
            cache['A' + str(i + 1)] = (tanh / keep_prob) * D
            cache['D' + str(i + 1)] = D

    return cache
