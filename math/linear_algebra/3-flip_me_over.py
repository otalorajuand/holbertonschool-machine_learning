#!/usr/bin/env python3
"""This module contains the function matrix_transpose."""


def matrix_transpose(matrix):
    """This functions transpose a 2D matrix"""
    res = []
    for column in range(len(matrix[0])):
        aux_row = []
        for row in matrix:
            aux_row.append(row[column])

        res.append(aux_row)
    return res
