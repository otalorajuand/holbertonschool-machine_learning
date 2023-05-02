#!/usr/bin/env python3
"""This module contains the function l2_reg_gradient_descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using
       gradient descent with L2 regularization

    Params:
        Y: one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
        weights: a dictionary of the weights and biases of the nn
        cache: is a dictionary of the outputs of each layer of the nn
        alpha: the learning rate
        lambtha: the L2 regularization parameter
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
    DW += (lambtha / m) * weights_copy['W' + str(nl)]
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

        deltas['DZ' + str(i)] = np.matmul(W.T, DZ) * (1 - (A * A))
        DZ_1 = deltas['DZ' + str(i)]
        DW = ((1 / m) * np.matmul(A_1, DZ_1.T)) + \
            (lambtha / m) * weights_copy['W' + str(i)].T
        DB = (1 / m) * np.sum(DZ_1, axis=1, keepdims=True)

        W = 'W' + str(i)
        B = 'b' + str(i)
        weights[W] = weights_copy[W] - (alpha * DW).T
        weights[B] = weights_copy[B] - alpha * DB
