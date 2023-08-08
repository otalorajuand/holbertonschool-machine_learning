#!/usr/bin/env python3
"""This module contains the class GRUCell"""
import numpy as np


class GRUCell:
    """represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """Class constructor

        Args:
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs

        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
        """

        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """performs forward propagation for one time step

        Args:
        - x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
          * m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state

        Returns: h_next, y
        - h_next is the next hidden state
        - y is the output of the cell
        """
        z_next = self.sigmoid(
            np.concatenate(
                (h_prev, x_t), axis=1) @ self.Wz + self.bz)
        r_next = self.sigmoid(
            np.concatenate(
                (h_prev, x_t), axis=1) @ self.Wr + self.br)
        c_next = np.tanh(
            np.concatenate(
                (h_prev *
                 r_next,
                 x_t),
                axis=1) @ self.Wh +
            self.bh)
        h_next = (1 - z_next) * h_prev + z_next * c_next

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y