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

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A = np.where(self.forward_prop(X)[0] < 0.5, 0, 1)
        return A, self.cost(Y, self.forward_prop(X)[0])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        deltas = {}
        weights = self.__weights.copy()
        nl = self.L
        deltas['DZ' + str(nl)] = cache['A' + str(nl)] - Y
        m = Y.shape[1]
        DW = (1 / m) * np.dot(deltas['DZ'+str(nl)], cache['A'+str(nl - 1)].T)
        DB = (1 / m) * np.sum(deltas['DZ' + str(nl)], axis=1, keepdims=True)

        W = 'W' + str(nl)
        B = 'b' + str(nl)
        self.__weights[W] = weights[W] - alpha * DW
        self.__weights[B] = weights[B] - alpha * DB

        for i in reversed(range(1, nl)):

            W = weights['W' + str(i + 1)]
            DZ = deltas['DZ' + str(i + 1)]

            A = cache['A' + str(i)]
            A_1 = cache['A' + str(i - 1)]

            deltas['DZ' + str(i)] = np.matmul(W.T, DZ) * (A * (1 - A))
            DZ_1 = deltas['DZ' + str(i)]
            DW = (1 / m) * np.matmul(A_1, DZ_1.T)
            DB = (1 / m) * np.sum(DZ_1, axis=1, keepdims=True)

            W = 'W' + str(i)
            B = 'b' + str(i)
            self.__weights[W] = weights[W] - (alpha * DW).T
            self.__weights[B] = weights[B] - alpha * DB
