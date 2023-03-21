#!/usr/bin/env python3
"""This module contains the function mat_mul"""


def mat_mul(mat1, mat2):
    """This function multiplies two 2D matrices"""
    if len(mat1[0]) != len(mat2):
        return None
    res = []
    for row in mat1:
        new_row = []
        for col in range(len(mat2[0])):
            number = 0
            for num1, num2 in zip(row, [row[col] for row in mat2]):
                number += num1 * num2
            new_row.append(number)
        res.append(new_row)
    return res
