#!/usr/bin/env python3
"""This module containts the function add_matrices"""


def add_matrices(mat1, mat2):
    """This function adds two matrices"""
    if not isinstance(mat1, type(mat2)):
        return None
    if isinstance(mat1, list):
        if len(mat1) != len(mat2):
            return None
        return [add_matrices(sub1, sub2) for sub1, sub2 in zip(mat1, mat2)]
    else:
        return mat1 + mat2
