#!/usr/bin/env python3
"""This module contains the function np_cat"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """This functions concatenates two matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
