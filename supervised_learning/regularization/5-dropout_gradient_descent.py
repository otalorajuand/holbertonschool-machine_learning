#!/usr/bin/env python3
"""This module includes the function dropout_gradient_descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout
    regularization using gradient descent

    Params:
        Y: one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
        weights: a dictionary of the weights and biases of the nn
        cache: is a dictionary of the outputs of each layer of the nn
        alpha: the learning rate
        keep_prob: the probability that a node will be kept
        L: the number of layers of the network
    
    Returns: Nothing
    """

    deltas = {}
    nl = L
    weights_copy = weights.copy()
    deltas['DZ' + str(nl)] = cache['A' + str(nl)] - Y
    m = Y.shape[1]
    DW = (1 / m) * np.dot(deltas['DZ' + str(nl)], cache['A' +
                                                        str(nl - 1)].T)
    DB = (1 / m) * np.sum(deltas['DZ' + str(nl)], axis=1, keepdims=True)

    W = 'W' + str(nl)
    B = 'b' + str(nl)
    weights[W] = weights_copy[W] - alpha * DW
    weights[B] = weights_copy[B] - alpha * DB

    for i in reversed(range(1, nl)):

        W = weights_copy['W' + str(i + 1)]
        DZ = deltas['DZ' + str(i + 1)]

        A = cache['A' + str(i)]
        A_1 = cache['A' + str(i - 1)]

        deltas['DZ' + str(i)] = np.matmul(W.T, DZ) * (A * (1 - A))
        deltas['DZ' + str(i)] /= keep_prob
        DZ_1 = deltas['DZ' + str(i)]
        DW = ((1 / m) * np.matmul(A_1, DZ_1.T))
        DB = (1 / m) * np.sum(DZ_1, axis=1, keepdims=True)

        W = 'W' + str(i)
        B = 'b' + str(i)
        weights[W] = weights_copy[W] - (alpha * DW).T
        weights[B] = weights_copy[B] - alpha * DB