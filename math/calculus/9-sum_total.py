#!/usr/bin/env python3
"""This module contains the function summation_i_squared"""


def summation_i_squared(n):
    """This function calculates the sum of the numbers squared
    until n"""

    summation = 0
    for i in range(n + 1):
        summation += i ** 2
    return summation
