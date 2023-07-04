#!/usr/bin/env python3
"""This module contains the function likelihood"""
import numpy as np
from scipy.special import comb


def likelihood(x, n, P):
    """calculates the likelihood of obtaining this data given
       various hypothetical probabilities of developing severe side effects.

    Params:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: a 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects

    Returns: a 1D numpy.ndarray containing the likelihood of obtaining
             the data, x and n, for each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray):
        TypeError('P must be a 1D numpy.ndarray')

    if np.all(np.logical_or(P < 0, P > 1)):
        ValueError('All values in P must be in the range [0, 1]')

    C = comb(n, x)

    return C * (P ** x) * ((1 - P) ** (n - x))
