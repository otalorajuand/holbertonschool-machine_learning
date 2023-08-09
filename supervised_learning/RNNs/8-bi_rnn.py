#!/usr/bin/env python3
"""This module contains the function bi_rnn"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for a bidirectional RNN

    Args:
    - bi_cell is an instance of BidirectinalCell that will be used for
      the forward propagation
    - X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        * t is the maximum number of time steps
        * m is the batch size
        * i is the dimensionality of the data
    - h_0 is the initial hidden state in the forward direction, given as a
      numpy.ndarray of shape (m, h)
        * h is the dimensionality of the hidden state
    - h_t is the initial hidden state in the backward direction, given as a
      numpy.ndarray of shape (m, h)

    Returns: H, Y
    - H is a numpy.ndarray containing all of the concatenated hidden states
    - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape

    H_F = np.zeros(shape=(t + 1, m, h))
    H_B = np.zeros(shape=(t + 1, m, h))

    H_F[0] = h_0
    H_B[t] = h_t

    for it in range(t):
        H_F[it + 1] = bi_cell.forward(H_F[it], X[it])

    for it in reversed(range(t)):
        H_B[it] = bi_cell.backward(H_B[it + 1], X[it])

    H = np.concatenate((H_F[1:], H_B[:t]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
