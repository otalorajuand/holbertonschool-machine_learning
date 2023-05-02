#!/usr/bin/env python3
"""This module contains the function l2_reg_cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization

    Params:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: a dictionary of the weights and biases (numpy.ndarrays)
        of the neural network
        L: the number of layers in the neural network
        m: the number of data points used

    Returns: the cost of the network accounting for L2 regularization
    """
    weights_l2 = sum([np.linalg.norm(v) for k, v in weights.items()
                      if k[0]='W'])
    cost_l2 = cost + ((lambtha / (2 * m)) * weights_l2)

    return cost_l2
