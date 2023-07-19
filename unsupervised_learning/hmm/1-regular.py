#!/usr/bin/env python3
"""This module contains the function regular"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov chain.
    Args:
        P: numpy.ndarray of shape (n, n) representing the transition matrix
           P[i, j] is the probability of transitioning from state i to state j
        n: number of states in the Markov chain
    Returns:
        numpy.ndarray of shape (1, n) containing the steady state probabilities,
        or None on failure
    """
    if not isinstance(
            P, np.ndarray) or len(
            P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None

    if np.any(P <= 0):
        return None

    n = P.shape[0]

    # Check if the matrix is regular (all rows must sum to 1)
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # Find eigenvalues and eigenvectors of the transition matrix
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the index of the eigenvalue that is equal to 1 (steady state)
    idx = np.where(np.isclose(eigenvalues, 1))[0]

    if len(idx) == 0:
        return None

    # Get the corresponding eigenvector
    steady_state_probabilities = np.real(eigenvectors[:, idx]).T

    return steady_state_probabilities / steady_state_probabilities.sum()
