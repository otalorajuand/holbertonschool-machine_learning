#!/usr/bin/env python3
"""This module contains the function summation_i_squared"""


def summation_i_squared(n):
    """This function calculates the sum of the numbers squared
    until n"""
    if not isinstance(n, int) and not isinstance(n, float):
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
