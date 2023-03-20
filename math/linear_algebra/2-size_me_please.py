#!/usr/bin/env python3


def matrix_shape(matrix):

    res = []
    block = matrix
    while True:
        res.append(len(block))
        block = block[0]
        if not isinstance(block, list):
            break
    return res
