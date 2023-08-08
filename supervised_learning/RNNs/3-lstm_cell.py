#!/usr/bin/env python3
"""This module contains the class LSTMCell"""
import numpy as np


class LSTMCell:
    """represents an LSTM unit"""

    def __init__(self, i, h, o):
        """Class constructor

        Args:
        - i is the dimensionality of the data
        - h is the dimensionality of the hidden state
        - o is the dimensionality of the outputs

        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
        """

        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """performs forward propagation for one time step

        Args:
        - x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
          * m is the batche size for the data
        - h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state
        - c_prev is a numpy.ndarray of shape (m, h) containing the previous
          cell state

        Returns: h_next, c_next, y
        - h_next is the next hidden state
        - c_next is the next cell state
        - y is the output of the cell
        """

        f_next = self.sigmoid(
            np.concatenate((h_prev, x_t), axis=1) @ self.Wf + self.bf)

        u_next = self.sigmoid(
            np.concatenate((h_prev, x_t), axis=1) @ self.Wu + self.bu)

        o_next = self.sigmoid(
            np.concatenate((h_prev, x_t), axis=1) @ self.Wo + self.bo)

        c_next_comma = np.tanh(
            np.concatenate((h_prev, x_t), axis=1) @ self.Wc + self.bc)

        c_next = f_next * c_prev + u_next * c_next_comma
        h_next = o_next * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
