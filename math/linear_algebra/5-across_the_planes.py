#!/usr/bin/env python3
"""this module contains the function add_matrices2D"""


def add_matrices2D(mat1, mat2):
    """this function add two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[a1 + a2 for a1, a2 in zip(arr1, arr2)]
            for arr1, arr2 in zip(mat1, mat2)]
