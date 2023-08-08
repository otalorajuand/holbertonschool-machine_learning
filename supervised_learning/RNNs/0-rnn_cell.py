#!/usr/bin/env python3
"""This module contains the class RNNCell"""
import numpy as np


class RNNCell:
    """represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """Class constructor

        Args:
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs

        Creates the public instance attributes Wh, Wy, bh, by
        that represent the weights and biases of the cell
        """

        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step

        Args:
        - x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
        - h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state

        Returns: h_next, y
        - h_next is the next hidden state
        - y is the output of the cell
        """
        def softmax(x):
            return (np.exp(x) / np.exp(x).sum())

        h_next = np.tanh(np.matmul(np.concatenate((h_prev, x_t), axis=1),
                         self.Wh) +
                         self.bh)

        y = softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
