#!/usr/bin/env python3
"""This module contains the class Neuron"""
import numpy as np


class Neuron:
    """This class defines a Neuron"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """This function Calculates the forward propagation of the neuron"""
        Z = np.dot(self.W, X) + self.b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = -(np.sum(loss) / loss.shape[1])
        return cost
