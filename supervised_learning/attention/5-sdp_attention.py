#!/usr/bin/env python3
"""This module contains the function sdp_attention"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention

    Args:
    - Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
      containing the query matrix
    - K is a tensor with its last two dimensions as (..., seq_len_v, dk)
      containing the key matrix
    - V is a tensor with its last two dimensions as (..., seq_len_v, dv)
      containing the value matrix
    - mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
      containing the optional mask, or defaulted to None

    Returns: output, weights
    - outputa tensor with its last two dimensions as (..., seq_len_q, dv)
      containing the scaled dot product attention
    - weights a tensor with its last two dimensions as (..., seq_len_q, seq_len_v)
      containing the attention weights
    """
    seq_len_q, dk = tf.shape(Q)[-2], float(tf.shape(Q)[-1])
    seq_len_v = tf.shape(K)[-2]
    dv = tf.shape(V)[-1]

    qk_matmul = tf.matmul(Q, K, transpose_b=True)

    scaled = qk_matmul / tf.math.sqrt(dk)

    if mask is not None:
        mask = mask * -1e9
        scaled = scaled + mask

    weights = tf.nn.softmax(scaled)

    output = tf.matmul(weights, V)

    return output, weights
