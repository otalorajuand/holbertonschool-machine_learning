#!/usr/bin/env python3
"""This module containts the function matrix_shape"""


def matrix_shape(matrix):
    """This function calculates the dimesions of a matrix and
        returns it in a list of integers"""
    res = []
    block = matrix
    while True:
        res.append(len(block))
        block = block[0]
        if not isinstance(block, list):
            break
    return res
