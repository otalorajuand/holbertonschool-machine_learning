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
        if [x for x in layers if x <= 0] != []:
            raise ValueError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(len(layers)):
            if i == 0:
                self.weights[f'W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f'W{i+1}'] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))
