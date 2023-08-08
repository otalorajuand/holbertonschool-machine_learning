#!/usr/bin/env python3
"""This module contains the function rnn"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN

    Args:
    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      * t is the maximum number of time steps
      * m is the batch size
      * i is the dimensionality of the data

    - h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        * h is the dimensionality of the hidden state

    Returns: H, Y
      - H is a numpy.ndarray containing all of the hidden states
      - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    m, h = h_0.shape
    h_next = h_0
    o = rnn_cell.by.shape[1]

    H = np.empty((t + 1, m, h))
    H[0, :, :] = h_0
    h_next = h_0
    Y = np.empty((t, m, o))

    for iter in range(t):

        h_next, y = rnn_cell.forward(h_next, X[iter, :, :])

        H[iter + 1, :, :] = h_next
        Y[iter, :, :] = y

    return H, Y
