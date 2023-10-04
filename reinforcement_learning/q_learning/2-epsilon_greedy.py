#!/usr/bin/env python3
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the next action

    Args:
    - Q is a numpy.ndarray containing the q-table
    - state is the current state
    - epsilon is the epsilon to use for the calculation

    Returns: the next action index
    """

    p = np.random.uniform()

    if p > epsilon:
        next_action = np.argmax(Q[state, :])
    else:
        next_action = np.random.randint(0, Q.shape[1])

    return next_action
