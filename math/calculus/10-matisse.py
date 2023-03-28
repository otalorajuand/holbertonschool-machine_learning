#!/usr/bin/env python3
"""This module contains the function poly_derivative"""


def poly_derivative(poly):
    """This function calculates the derivative of poly"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    return [(index + 1) * elem for index, elem in enumerate(poly[1:])]
