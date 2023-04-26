#!/usr/bin/env python3
"""This module contains the function batch_norm"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output of a neural network
    using batch normalization"""
    miu = np.mean(Z, axis=0)
    G2 = np.std(Z, axis=0) ** 2

    Z_norm = (Z - miu) / ((G2 + epsilon)**0.5)
    return gamma * Z_norm + beta
