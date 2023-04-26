#!/usr/bin/env python3
"""This module contains the function update_variables_Adam"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm"""
    vd = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    vd_corrected = vd/(1 - (beta1 ** t)) 
    s_corrected = s/(1 - (beta2 ** t))
    var = var - alpha * vd_corrected / ((s_corrected ** 0.5) + epsilon)
    return var, vd_corrected, s_corrected
