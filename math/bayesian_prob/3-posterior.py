#!/usr/bin/env python3
"""This module contains the function posterior"""

intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """calculates the posterior probability for the various hypothetical
       probabilities of developing severe side effects given the data.

    Params:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: a 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects
        Pr: a 1D numpy.ndarray containing the prior beliefs of P

    Returns: the posterior probability of each probability in
             P given x and n, respectively
    """
    inter = intersection(x, n, P, Pr)
    marg = marginal(x, n, P, Pr)

    return inter / marg
