#!/usr/bin/env python3
"""This module contains the function deep_rnn"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep RNN

    Args:
    - rnn_cells is a list of RNNCell instances of length l that will be used
      for the forward propagation
      * l is the number of layers

    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
      * t is the maximum number of time steps
      * m is the batch size
      * i is the dimensionality of the data

    - h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
        * h is the dimensionality of the hidden state

    Returns: H, Y
      - H is a numpy.ndarray containing all of the hidden states
      - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    l, m, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.empty((t + 1, l, m, h))
    H[0, :, :, :] = h_0
    h_next = h_0
    Y = np.empty((t, m, o))

    for iter in range(t):

        h_next, y = rnn_cells[0].forward(H[iter, 0], X[iter])

        for cell in range(1, l):

            h_next, y = rnn_cells[cell].forward(H[iter, cell], h_next)

            H[iter + 1, cell, :, :] = h_next

        Y[iter, :, :] = y

    return H, Y
