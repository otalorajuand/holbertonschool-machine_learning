#!/usr/bin/env python3
"""This module includes the function conv_backward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform backpropagation over a convolutional layer.

    Arguments:
    dZ -- numpy.ndarray of shape (m, h_new, w_new, c_new) containing the partial derivatives
          with respect to the unactivated output of the convolutional layer
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing the output of
              the previous layer
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for the convolution
    b -- numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    padding -- string, type of padding used: 'same' or 'valid' (default: 'same')
    stride -- tuple of (sh, sw) containing the strides for the convolution
              sh is the stride for the height
              sw is the stride for the width

    Returns:
    dA_prev -- partial derivatives with respect to the previous layer
    dW -- partial derivatives with respect to the kernels
    db -- partial derivatives with respect to the biases
    """
    
    # Retrieve dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    
    # Initialize output gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    
    # Apply padding to dA_prev based on the padding type
    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
        dA_prev = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    elif padding == "valid":
        pad_h = 0
        pad_w = 0
    
    # Backpropagation loop
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # Compute gradients for dA_prev, dW, db
                    dA_prev[i, h * sh:h * sh + kh, w * sw:w * sw + kw, :] += np.sum(
                        W[:, :, :, c] * dZ[i, h, w, c], axis=-1)
                    dW[:, :, :, c] += A_prev[i, h * sh:h * sh + kh, w * sw:w * sw + kw, :] * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
    
    # Adjust dA_prev if padding is used
    if padding == "same":
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    
    return dA_prev, dW, db
