#!/usr/bin/env python3
"""This module includes the function policy"""
import numpy as np


def policy(matrix, weight):
    """computes to policy with a weight of a matrix
    """

    x = matrix @ weight
    return np.exp(x) / np.exp(x).sum()


def policy_gradient(state, weight):
    """computes the Monte-Carlo policy gradient based on a state
       and a weight matrix

    Args:
    - state: matrix representing the current observation of the environment
    - weight: matrix of random weight

    Return: the action and the gradient (in this order)
    """
    probs = policy(state, weight)
    
    # Choose an action randomly based on the probabilities
    action = np.random.choice([0, 1], p=probs[0])
    
    # Compute the gradient of the log-probability of the chosen action
    grad = state.T - probs
    grad = grad * (1 - probs)
    grad = grad[action]

    return action, grad
