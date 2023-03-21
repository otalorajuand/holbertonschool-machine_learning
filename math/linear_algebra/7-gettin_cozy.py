#!/usr/bin/env python3
"""This module contains the function cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """This function concateantes two 2D matrices"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        mat1_copy, mat2_copy = mat1.copy(), mat2.copy()
        return [*mat1_copy, *mat2_copy]
    else:
        if len(mat1) != len(mat2):
            return None
        return [[*row1, *row2] for row1, row2 in zip(mat1, mat2)]

