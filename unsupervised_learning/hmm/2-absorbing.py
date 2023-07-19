#!/usr/bin/env python3
"""This module contains the function absorbing"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing

    Params:
        P: a is a square 2D numpy.ndarray of shape (n, n) representing the
           standard transition matrix
            P[i, j]: the probability of transitioning from state i to state j
            n: the number of states in the markov chain

    Returns: True if it is absorbing, or False on failure
    """

    if not isinstance(
            P, np.ndarray) or len(
            P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False

    if np.any(P == 1):
        return True

    return False
