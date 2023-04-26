#!/usr/bin/env python3
"""This module contains the function moving_average"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    vt = 0
    res = []

    for i, elem in enumerate(data):
        vt = beta*vt + (1-beta)*elem
        res.append(vt/(1-(beta**(i+1))))

    return res

