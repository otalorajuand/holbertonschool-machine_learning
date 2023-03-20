#!/usr/bin/env python3
"""This module containts the function add_arrays"""


def add_arrays(arr1, arr2):
    """this functions adds up two lists element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [a1 + a2 for a1, a2 in zip(arr1, arr2)]
