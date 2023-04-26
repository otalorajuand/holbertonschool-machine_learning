#!/usr/bin/env python3
"""this module contains the function update_variables_RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm"""
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * grad / ((s ** 0.5) + epsilon)
    return var, s
