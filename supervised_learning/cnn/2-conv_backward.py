#!/usr/bin/env python3
"""This module includes the function conv_backward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    dA_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    dA_prev = dA_prev_pad[:, pad_h:h_prev+pad_h, pad_w:w_prev+pad_w, :]

    return dA_prev, dW, db
