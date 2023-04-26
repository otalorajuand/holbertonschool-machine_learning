#!/usr/bin/env python3
"""This module contains the function update_variables_momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates a variable using the gradient descent with momentum optimization algorithm"""
    vd = beta1*v + (1-beta1)*grad
    var = var - alpha*vd
    return var, vd
