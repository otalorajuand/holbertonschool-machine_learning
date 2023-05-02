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

    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        dA = np.dot(W.T, dZ)
        if i > 1:
            dA *= cache["D" + str(i - 1)]
        dA /= keep_prob

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dZ = dA * (1 - np.power(A_prev, 2))

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
