#!/usr/bin/env python3
"""This module contains the class Normal"""


class Normal:
    """This class represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = (sum((x - (sum(data) / len(data))) **
                           2 for x in data) / len(data))**0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean)/self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean
