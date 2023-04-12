#!/usr/bin/env python3
"""This module includes the class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list) or layers == []:
            raise TypeError('layers must be a list of positive integers')
        if list(filter(lambda x: x <= 0, layers)) != []:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(len(layers)):
            if i == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """This function Calculates the forward propagation
        of the neural network"""

        self.__cache['A0'] = X

        for i in range(self.L):
            W = self.weights['W' + str(i + 1)]
            A = self.cache['A' + str(i)]
            B = self.weights['b' + str(i + 1)]

            Z = np.dot(W, A) + B

            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.cache['A' + str(i + 1)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using neural network"""
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(np.sum(loss) / loss.shape[1])
        return cost
