#!/usr/bin/env python3
"""This module contains the class BidirectionalCell"""
import numpy as np


class BidirectionalCell:
    """represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """Class constructor

        Args:
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden states
        - o is the dimensionality of the outputs

        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
        that represent the weights and biases of the cell
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """calculates the hidden state in the forward direction for one
        time step.

        Args:
        - x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
            * m is the batch size for the data

        - h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state

        Returns: h_next, the next hidden state
        """

        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh((concat @ self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """calculates the hidden state in the backward direction for one
           time step

        Args:
        - x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
          * m is the batch size for the data
        - h_next is a numpy.ndarray of shape (m, h) containing the next
          hidden state

        Returns: h_pev, the previous hidden state
        """

        concat = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh((concat @ self.Whb) + self.bhb)

        return h_pev

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def output(self, H):
        """calculates all outputs for the RNN

        Args:
        - H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
          concatenated hidden states from both directions, excluding their
          initialized states
          * t is the number of time steps
          * m is the batch size for the data
          * h is the dimensionality of the hidden states

        Returns: Y, the outputs
        """
        t, m, _ = H.shape
        o = self.by.shape[1]

        y = np.empty((t, m, o))

        for iter in range(t):
            output = self.softmax((H[iter] @ self.Wy) + self.by)
            y[iter] = output

        return y
