#!/usr/bin/env python3
"""This module contains the function poly_integral"""


def poly_integral(poly, C=0):
    """This function calculates the integral of poly"""
    if poly == 0:
        return [0]
    if not isinstance(poly, list):
        return None
    res = [C, *[elem / (index + 1) for index, elem in enumerate(poly)]]
    return [int(x) if isinstance(x, float)
            and x.is_integer() else x for x in res]
