#!/usr/bin/env python3
"""This module contains the function pool_backward"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network

    Params:
        - dA: a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
          the partial derivatives with respect to the output of the pooling
          layer
        - A_prev: a numpy.ndarray of shape (m, h_prev, w_prev, c)
          containing the output of the previous layer
        - kernel_shape is a tuple of (kh, kw) containing the size of the
          kernel for the pooling
        - stride is a tuple of (sh, sw) containing the strides for the pooling
        - mode is a string containing either max or avg, indicating whether
          to perform maximum or average pooling, respectively

    Returns: the partial derivatives with respect to the previous layer 
    (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros((m, h_prev, w_prev, c))

    for x in range(h_new):
        for y in range(w_new):
            for c in range(c):
                for example in range(m):
                    if mode == 'max':
                        a_prev_slice = A_prev[example, x: x + kh, y: y+kw, c]
                        mask = (a_prev_slice==np.max(a_prev_slice))
                        res = mask*dA[example, x, y, c]
                    elif mode == 'avg':
                        res = dA[example, x, y, c]/(kh*kw)

                    dA_prev[example, x: x + kh, y: y+kw, c] += res

    return dA_prev

