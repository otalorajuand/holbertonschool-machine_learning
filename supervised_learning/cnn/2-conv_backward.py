#!/usr/bin/env python3
"""This module includes the function conv_backward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of a neural network

    Params:
        - dZ: a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
          the partial derivatives with respect to the unactivated output
          of the convolutional layer
        - A_prev: a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
        - W: a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
          the kernels for the convolution
        - b: a numpy.ndarray of shape (1, 1, 1, c_new) containing the
          biases applied to the convolution
        - padding: a string that is either same or valid, indicating
          the type of padding used
        - stride: a tuple of (sh, sw) containing the strides for
          the convolution

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    pad_h, pad_w = 0, 0

    if padding == 'same':
        pad_h = ((((h_prev - 1) * sh) + kh - h_prev) // 2) + 1
        pad_w = ((((w_prev - 1) * sw) + kw - w_prev) // 2) + 1

    dA_prev = np.zeros((m, h_prev + (2 * pad_h), w_prev + (2 * pad_w), c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    output_pad = np.pad(A_prev, pad_width=((0, 0), (pad_h, pad_h),
                                           (pad_w, pad_w), (0, 0)),
                        mode='constant')
    

    for x in range(h_new):
        for y in range(w_new):
            for k in range(c_new):
                for example in range(m):
                    i = x * sh
                    j = y * sw
                    dA_prev[example, i:i + kh,
                            j:j + kw,
                            :] += dZ[example, x, y, k] * W[:, :, :, k]

                    dW[:, :, :, k] += output_pad[example,i:i + kh,j:j + kw,
                                                 :] * dZ[example, x, y, k]


    if padding == 'same':
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
